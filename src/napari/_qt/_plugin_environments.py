"""Qt startup and monitoring UI for managed plugin environments."""

from __future__ import annotations

import logging
from itertools import groupby
from typing import TYPE_CHECKING, Any

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

from napari.plugins._environment_manager import (
    _EnvironmentView,
    _PluginLogRecord,
    _SetupState,
    _WorkerState,
    get_plugin_environment_manager,
)
from napari.plugins.environments import (
    PluginTaskState,
    _add_task_observer,
    _set_task_dispatcher,
)
from napari.utils.notifications import notification_manager

if TYPE_CHECKING:
    from collections.abc import Callable

    from qtpy.QtGui import QCloseEvent

    from napari.plugins.environments import PluginTask

logger = logging.getLogger(__name__)
_installed = False
_startup_scheduled = False
_setup_dialog: _PluginSetupDialog | None = None


def _dispatch_to_main_thread(callback: Callable[[], None]) -> None:
    ensure_main_thread(callback)()


@ensure_main_thread
def _notify_task_failure(task: PluginTask[Any]) -> None:
    """Present a worker failure when plugin code does not handle it."""

    def receive_done(done_task: PluginTask[Any]) -> None:
        if (
            done_task.state is PluginTaskState.FAILED
            and done_task.error is not None
        ):
            error = done_task.error
            notification_manager.receive_error(
                type(error), error, error.__traceback__
            )

    task.add_done_callback(receive_done)


def _shutdown_with_notification() -> None:
    from napari.plugins._environment_manager import (
        _shutdown_plugin_environments_once,
    )

    try:
        _shutdown_plugin_environments_once()
    except Exception as error:  # noqa: BLE001
        notification_manager.receive_error(
            type(error), error, error.__traceback__
        )


def _format_log(record: _PluginLogRecord) -> str:
    scope = ' / '.join(
        value
        for value in (record.plugin, record.environment_id)
        if value is not None
    )
    prefix = record.timestamp.astimezone().strftime('%H:%M:%S')
    if scope:
        prefix = f'{prefix} | {scope}'
    if record.level != 'info':
        prefix = f'{prefix} | {record.level.upper()}'
    return f'{prefix} | {record.message}'


class _PluginSetupDialog(QDialog):
    """Application-modal progress for startup environment reconciliation."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Setting up plugin environments')
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(680, 440)
        self._manager = get_plugin_environment_manager()
        self._allow_close = False

        self._label = QLabel('Checking plugin environments…', self)
        self._progress = QProgressBar(self)
        self._logs = QPlainTextEdit(self)
        self._logs.setObjectName('plugin_environment_log')
        self._logs.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._logs.setReadOnly(True)
        self._buttons = QDialogButtonBox(self)
        self._retry = self._buttons.addButton(
            'Retry', QDialogButtonBox.ButtonRole.ActionRole
        )
        self._continue = self._buttons.addButton(
            'Continue without affected plugin workers',
            QDialogButtonBox.ButtonRole.AcceptRole,
        )
        self._quit = self._buttons.addButton(
            'Quit napari', QDialogButtonBox.ButtonRole.RejectRole
        )
        self._retry.clicked.connect(self._retry_setup)
        self._continue.clicked.connect(self._continue_setup)
        self._quit.clicked.connect(self._quit_napari)

        layout = QVBoxLayout(self)
        layout.addWidget(self._label)
        layout.addWidget(self._progress)
        layout.addWidget(self._logs, 1)
        layout.addWidget(self._buttons)

        self._manager.add_attention_callback(
            lambda: _dispatch_to_main_thread(self._show_for_attention)
        )
        self._manager.add_state_callback(
            lambda: _dispatch_to_main_thread(self._refresh)
        )
        self._manager.add_log_callback(
            lambda record: _dispatch_to_main_thread(
                lambda: self._append_log(record)
            )
        )
        self._refresh()

    def _show_for_attention(self) -> None:
        if not self.isVisible():
            self.show()
        self.raise_()
        self.activateWindow()

    def _append_log(self, record: _PluginLogRecord) -> None:
        self._logs.appendPlainText(_format_log(record))

    def _refresh(self) -> None:
        views = self._manager.environment_views()
        total = len(views)
        ready = sum(view.setup_state is _SetupState.READY for view in views)
        active = next(
            (
                view
                for view in views
                if view.setup_state is _SetupState.SETTING_UP
            ),
            None,
        )
        self._progress.setRange(0, max(total, 1))
        self._progress.setValue(ready)
        if active is not None:
            self._label.setText(
                f'Setting up {active.display_name} ({ready + 1} of {total})'
            )
        elif self._manager.setup_has_failures():
            self._label.setText(
                'Some plugin environments could not be set up.'
            )
        elif total:
            self._label.setText(f'{ready} of {total} environments are ready.')
        else:
            self._label.setText('No plugin environments are declared.')

        running = self._manager.setup_running()
        failed = self._manager.setup_has_failures()
        self._retry.setVisible(failed and not running)
        self._continue.setVisible(failed and not running)
        self._quit.setVisible(running or failed)
        if not running and not failed:
            self._allow_close = True
            self.hide()

    def _retry_setup(self) -> None:
        self._manager.retry_setup()
        self._refresh()

    def _continue_setup(self) -> None:
        self._manager.continue_without_failed()
        self._allow_close = True
        self.hide()

    def _quit_napari(self) -> None:
        self._manager.cancel_setup()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()
        self._quit_napari()


class _PluginWorkerCard(QFrame):
    """One managed environment and its process-local worker state."""

    def __init__(
        self,
        view: _EnvironmentView,
        *,
        stop: Callable[[str], None],
        show_logs: Callable[[str], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.environment_id = view.environment_id
        self._stop = stop
        self._show_logs = show_logs
        self.setObjectName('plugin_worker_card')
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum
        )

        self._title = QLabel(self)
        self._title.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._identifier = QLabel(self)
        self._identifier.setObjectName('plugin_worker_secondary')
        self._identifier.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._status = QLabel(self)
        self._status.setObjectName('plugin_worker_secondary')
        self._diagnostic = QLabel(self)
        self._diagnostic.setObjectName('plugin_worker_diagnostic')
        self._diagnostic.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._diagnostic.setWordWrap(True)

        self._show_logs_button = QPushButton('Show logs', self)
        self._show_logs_button.clicked.connect(
            lambda: self._show_logs(self.environment_id)
        )
        self._action_button = QPushButton('Stop worker', self)
        self._action_button.clicked.connect(
            lambda: self._stop(self.environment_id)
        )

        buttons = QHBoxLayout()
        buttons.addWidget(self._show_logs_button)
        buttons.addWidget(self._action_button)
        buttons.addStretch()

        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setVerticalSpacing(4)
        layout.addWidget(self._title, 0, 0)
        layout.addWidget(self._identifier, 0, 1)
        layout.addWidget(self._status, 1, 0, 1, 2)
        layout.addWidget(self._diagnostic, 2, 0, 1, 2)
        layout.addLayout(buttons, 3, 0, 1, 2)
        layout.setColumnStretch(0, 1)

        self.set_view(view)

    def set_view(self, view: _EnvironmentView) -> None:
        """Update visible state without replacing the card widgets."""

        self._title.setText(f'<b>{view.display_name}</b>')
        self._identifier.setText(view.environment_id)
        self._status.setText(
            f'Setup: {_state_label(view.setup_state)} · '
            f'Worker: {_state_label(view.worker_state)}'
        )
        self._diagnostic.setText(view.diagnostic or '')
        self._diagnostic.setVisible(bool(view.diagnostic))

        cleanup_failed = view.worker_state is _WorkerState.CLEANUP_FAILED
        self._action_button.setText(
            'Retry worker cleanup' if cleanup_failed else 'Stop worker'
        )
        self._action_button.setEnabled(
            cleanup_failed or view.worker_state is _WorkerState.IDLE
        )
        self._action_button.setToolTip(
            _worker_action_tooltip(view.worker_state)
        )


def _state_label(state: _SetupState | _WorkerState) -> str:
    return state.value.replace('_', ' ').title()


def _worker_action_tooltip(state: _WorkerState) -> str:
    if state is _WorkerState.IDLE:
        return 'Stop this idle worker to release its memory.'
    if state is _WorkerState.CLEANUP_FAILED:
        return 'Retry stopping a worker whose previous cleanup failed.'
    if state is _WorkerState.STOPPED:
        return 'This worker is already stopped.'
    return 'Only an idle plugin worker can be stopped.'


class PluginWorkersDialog(QDialog):
    """Read-only environment inventory with idle worker controls and logs."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle('Plugin Environments')
        self.resize(820, 600)
        self._manager = get_plugin_environment_manager()
        self._state_unsubscribe: Callable[[], None] | None = None
        self._log_unsubscribe: Callable[[], None] | None = None
        self._inventory_signature: tuple[tuple[str, ...], ...] = ()
        self._cards: dict[str, _PluginWorkerCard] = {}
        self._group_titles: list[QLabel] = []

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName('plugin_worker_list')
        self._scroll.setWidgetResizable(True)
        self._content = QWidget(self._scroll)
        self._content.setObjectName('plugin_worker_content')
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._empty = QLabel(
            'No managed plugin environments are declared for this napari '
            'session.',
            self._content,
        )
        self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty.setWordWrap(True)
        self._content_layout.addWidget(self._empty, 1)
        self._scroll.setWidget(self._content)

        self._filter = QComboBox(self)
        self._filter.currentIndexChanged.connect(self._rebuild_logs)
        self._copy_button = QPushButton('Copy', self)
        self._clear_button = QPushButton('Clear all logs', self)
        self._copy_button.clicked.connect(self._copy_logs)
        self._clear_button.clicked.connect(self._clear_logs)
        controls = QWidget(self)
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(QLabel('Logs:', controls))
        controls_layout.addWidget(self._filter, 1)
        controls_layout.addWidget(self._copy_button)
        controls_layout.addWidget(self._clear_button)

        self._logs = QPlainTextEdit(self)
        self._logs.setObjectName('plugin_environment_log')
        self._logs.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self._logs.setReadOnly(True)
        self._logs.setPlaceholderText(
            'Environment setup, worker execution, cancellation, and cleanup '
            'messages will appear here.'
        )
        log_widget = QWidget(self)
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(controls)
        log_layout.addWidget(self._logs, 1)

        splitter = QSplitter(Qt.Orientation.Vertical, self)
        splitter.addWidget(self._scroll)
        splitter.addWidget(log_widget)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([400, 200])

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Close, parent=self
        )
        self._close_button = buttons.button(
            QDialogButtonBox.StandardButton.Close
        )
        self._close_button.clicked.connect(self.close)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, 1)
        layout.addWidget(buttons)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._state_unsubscribe is None:
            self._state_unsubscribe = self._manager.add_state_callback(
                lambda: _dispatch_to_main_thread(self._refresh)
            )
            self._log_unsubscribe = self._manager.add_log_callback(
                lambda record: _dispatch_to_main_thread(
                    lambda: self._receive_log(record)
                ),
                replay=False,
            )
        self._refresh()
        self._rebuild_logs()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._state_unsubscribe is not None:
            self._state_unsubscribe()
            self._state_unsubscribe = None
        if self._log_unsubscribe is not None:
            self._log_unsubscribe()
            self._log_unsubscribe = None
        event.accept()

    def _refresh(self) -> None:
        views = self._manager.environment_views()
        ordered = tuple(
            sorted(
                views,
                key=lambda view: (
                    view.plugin_display_name.casefold(),
                    view.plugin.casefold(),
                    view.display_name.casefold(),
                    view.environment_id.casefold(),
                ),
            )
        )
        self._sync_cards(ordered)
        filter_changed = self._sync_filter(ordered)
        for view in ordered:
            self._cards[view.environment_id].set_view(view)
        if filter_changed:
            self._rebuild_logs()

    def _sync_cards(self, views: tuple[_EnvironmentView, ...]) -> None:
        signature = tuple(
            (
                view.plugin,
                view.plugin_display_name,
                view.environment_id,
                view.display_name,
            )
            for view in views
        )
        if signature == self._inventory_signature:
            return
        self._inventory_signature = signature

        while (item := self._content_layout.takeAt(0)) is not None:
            if (widget := item.widget()) is not None:
                widget.deleteLater()
        self._cards.clear()
        self._group_titles.clear()

        if not views:
            self._empty = QLabel(
                'No managed plugin environments are declared for this napari '
                'session.',
                self._content,
            )
            self._empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._empty.setWordWrap(True)
            self._content_layout.addWidget(self._empty, 1)
            return

        for _plugin, grouped_views in groupby(
            views, key=lambda view: view.plugin
        ):
            group = tuple(grouped_views)
            heading = QLabel(
                f'<b>{group[0].plugin_display_name}</b>', self._content
            )
            heading.setObjectName('plugin_worker_group_title')
            heading.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            self._group_titles.append(heading)
            self._content_layout.addWidget(heading)
            for view in group:
                card = _PluginWorkerCard(
                    view,
                    stop=self._stop,
                    show_logs=self._show_environment_logs,
                    parent=self._content,
                )
                self._cards[view.environment_id] = card
                self._content_layout.addWidget(card)
        self._content_layout.addStretch()

    def _sync_filter(self, views: tuple[_EnvironmentView, ...]) -> bool:
        selected = self._filter.currentData()
        desired = [('All environments', None)] + [
            (
                f'{view.plugin_display_name} — {view.display_name}',
                view.environment_id,
            )
            for view in views
        ]
        current = [
            (self._filter.itemText(index), self._filter.itemData(index))
            for index in range(self._filter.count())
        ]
        if current == desired:
            return False
        self._filter.blockSignals(True)
        self._filter.clear()
        for label, value in desired:
            self._filter.addItem(label, value)
        index = self._filter.findData(selected)
        self._filter.setCurrentIndex(max(index, 0))
        self._filter.blockSignals(False)
        return self._filter.currentData() != selected

    def _show_environment_logs(self, environment_id: str) -> None:
        index = self._filter.findData(environment_id)
        if index >= 0:
            self._filter.setCurrentIndex(index)
        self._logs.setFocus()

    def _stop(self, environment_id: str) -> None:
        try:
            self._manager.stop_worker(environment_id)
        except Exception as error:  # noqa: BLE001
            notification_manager.receive_error(
                type(error), error, error.__traceback__
            )

    def _record_matches(self, record: _PluginLogRecord) -> bool:
        selected = self._filter.currentData()
        return selected is None or record.environment_id == selected

    def _receive_log(self, record: _PluginLogRecord) -> None:
        if self._record_matches(record):
            scrollbar = self._logs.verticalScrollBar()
            at_bottom = scrollbar.value() >= scrollbar.maximum() - 1
            previous = scrollbar.value()
            self._logs.appendPlainText(_format_log(record))
            if at_bottom:
                scrollbar.setValue(scrollbar.maximum())
            else:
                scrollbar.setValue(previous)

    def _rebuild_logs(self) -> None:
        self._logs.setPlainText(
            '\n'.join(
                _format_log(record)
                for record in self._manager.log_records()
                if self._record_matches(record)
            )
        )
        scrollbar = self._logs.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _copy_logs(self) -> None:
        clipboard = QApplication.clipboard()
        clipboard.setText(self._logs.toPlainText())

    def _clear_logs(self) -> None:
        self._manager.clear_logs()
        self._logs.clear()


@ensure_main_thread
def schedule_plugin_environment_reconciliation() -> None:
    """Start the immutable reconciliation pass after the first window shows."""

    global _startup_scheduled, _setup_dialog
    if _startup_scheduled:
        return
    _startup_scheduled = True
    from napari._qt.qt_main_window import _QtMainWindow

    _setup_dialog = _PluginSetupDialog(_QtMainWindow.current())
    get_plugin_environment_manager().start_reconciliation()


def show_plugin_workers(parent: QWidget | None = None) -> PluginWorkersDialog:
    """Show or raise napari's process-local plugin worker monitor."""

    dialog = PluginWorkersDialog(parent)
    dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
    dialog.show()
    dialog.raise_()
    dialog.activateWindow()
    return dialog


def install_plugin_environment_qt_support(app: QApplication) -> None:
    """Install task dispatch, failure notification, and shutdown once."""

    global _installed
    if _installed:
        return
    _installed = True
    _set_task_dispatcher(_dispatch_to_main_thread)
    _add_task_observer(_notify_task_failure)
    app.aboutToQuit.connect(_shutdown_with_notification)


__all__ = (
    'PluginWorkersDialog',
    'install_plugin_environment_qt_support',
    'schedule_plugin_environment_reconciliation',
    'show_plugin_workers',
)
