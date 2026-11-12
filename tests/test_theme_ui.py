"""Regression tests for theme placement, UI styling and sidebar controls."""

from unittest.mock import MagicMock

import app


def _fake_streamlit(monkeypatch, *, light: bool) -> MagicMock:
    """Create a mock Streamlit UI with the selected session theme."""
    fake = MagicMock()
    fake.session_state = {"light_mode": light}
    fake.columns.return_value = (MagicMock(), MagicMock())
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_uses_two_full_width_buttons(monkeypatch) -> None:
    """Sun and moon actions occupy equal halves of one compact pill."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._theme_switch()
    fake.container.assert_any_call(key="theme_dock")
    fake.columns.assert_called_once_with(2, gap="small")
    assert fake.button.call_count == 2
    labels = [call.args[0] for call in fake.button.call_args_list]
    assert labels == ["☼", "☾"]
    assert all(call.kwargs["use_container_width"] for call in fake.button.call_args_list)
    fake.segmented_control.assert_not_called()


def test_theme_callbacks_change_session_state(monkeypatch) -> None:
    """Both icon buttons set the chosen mode without clearing notes."""
    fake = _fake_streamlit(monkeypatch, light=False)
    fake.session_state["messages"] = [{"role": "user", "content": "hello"}]
    app._set_theme(True)
    assert fake.session_state["light_mode"] is True
    app._set_theme(False)
    assert fake.session_state["light_mode"] is False
    assert fake.session_state["messages"][0]["content"] == "hello"


def test_active_half_is_selected(monkeypatch) -> None:
    """Active sun or moon gets its own CSS class."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._theme_switch()
    fake.container.assert_any_call(key="theme_sun_active")
    fake.container.assert_any_call(key="theme_moon")


def test_theme_dock_has_no_empty_space(monkeypatch) -> None:
    """Height is explicitly capped to one row of two buttons."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert 'section[data-testid="stSidebar"] .st-key-theme_dock' in css
    assert "width: 126px !important;" in css
    assert "max-height: 52px;" in css
    assert ".st-key-theme_sun_active button" in css
    assert ".sr-theme-label" not in css


def test_light_theme_injects_light_chat_and_sidebar_styles(monkeypatch) -> None:
    """Light mode changes the sidebar, message and background colors."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._styles()
    css = "\n".join(call.args[0] for call in fake.markdown.call_args_list)
    assert "#f6f8fd" in css
    assert "#ffffff" in css
    assert 'section[data-testid="stSidebar"]' in css
    assert 'data-testid="stChatMessage"' in css
    assert ".st-key-theme_sun_active button" in css


def test_dark_theme_does_not_include_light_overrides(monkeypatch) -> None:
    """Dark mode retains the night palette during every rerender."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    assert fake.markdown.call_count == 1
    css = fake.markdown.call_args.args[0]
    assert "#0b1020" in css
    assert "#f6f8fd" not in css


def test_native_sidebar_controls_remain_visible(monkeypatch) -> None:
    """Collapse and reopen controls must remain outside toolbar hiding."""
    fake = _fake_streamlit(monkeypatch, light=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert '[data-testid="stSidebarCollapsedControl"]' in css
    assert '[data-testid="stSidebarCollapseButton"]' in css
    assert '[data-testid="stToolbar"],' not in css
    assert '[data-testid="stAppDeployButton"]' in css


def test_sidebar_toggle_supports_light_theme(monkeypatch) -> None:
    """Sidebar expand/collapse buttons remain readable in light mode."""
    fake = _fake_streamlit(monkeypatch, light=True)
    app._styles()
    light_css = fake.markdown.call_args_list[-1].args[0]
    assert '[data-testid="stSidebarCollapsedControl"] button' in light_css
    assert '[data-testid="stSidebarCollapseButton"] button' in light_css
