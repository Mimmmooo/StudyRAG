"""Regression tests for the theme switch and rendered color schemes."""

from unittest.mock import MagicMock

import app


def _fake_streamlit(monkeypatch, *, light: bool, clicked: bool) -> MagicMock:
    """Create an in-memory Streamlit interface without opening a browser."""
    fake = MagicMock()
    fake.session_state = {"light_mode": light}
    fake.button.return_value = clicked
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_toggle_lives_inside_sidebar_container(monkeypatch) -> None:
    """Use Streamlit's native two-state toggle, not a floating button."""
    fake = _fake_streamlit(monkeypatch, light=False, clicked=False)
    app._theme_switch()
    fake.container.assert_called_once_with(key="theme_control")
    fake.toggle.assert_called_once()
    assert fake.toggle.call_args.kwargs["key"] == "light_mode"
    assert "☀️" in fake.toggle.call_args.args[0]
    assert "🌙" in fake.toggle.call_args.args[0]
    fake.button.assert_not_called()
    fake.rerun.assert_not_called()


def test_theme_toggle_does_not_overwrite_existing_state(monkeypatch) -> None:
    """The real Streamlit toggle owns the selected theme state."""
    fake = _fake_streamlit(monkeypatch, light=True, clicked=False)
    app._theme_switch()
    assert fake.session_state["light_mode"] is True
    fake.rerun.assert_not_called()


def test_theme_css_is_scoped_to_sidebar_not_top_right(monkeypatch) -> None:
    """Floating theme CSS must not collide with loading and toolbar icons."""
    fake = _fake_streamlit(monkeypatch, light=False, clicked=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert 'section[data-testid="stSidebar"] .st-key-theme_control' in css
    assert "position: fixed;\n          top: .85rem;" not in css
    assert 'data-testid="stToggle"' in css


def test_light_theme_injects_light_chat_and_sidebar_styles(monkeypatch) -> None:
    """Switching to light mode changes the surface colors, not just the icon."""
    fake = _fake_streamlit(monkeypatch, light=True, clicked=False)
    app._styles()
    css = "\n".join(
        call.args[0] for call in fake.markdown.call_args_list
    )
    assert "#f6f8fd" in css
    assert "#ffffff" in css
    assert 'section[data-testid="stSidebar"]' in css
    assert 'data-testid="stChatMessage"' in css


def test_dark_theme_does_not_include_light_overrides(monkeypatch) -> None:
    """Dark mode should keep the full dark palette on every rerender."""
    fake = _fake_streamlit(monkeypatch, light=False, clicked=False)
    app._styles()
    assert fake.markdown.call_count == 1
    css = fake.markdown.call_args.args[0]
    assert "#0b1020" in css
    assert "#f6f8fd" not in css


def test_native_sidebar_controls_remain_visible(monkeypatch) -> None:
    """Do not hide the built-in collapse and reopen buttons with toolbar CSS."""
    fake = _fake_streamlit(monkeypatch, light=False, clicked=False)
    app._styles()
    css = fake.markdown.call_args.args[0]
    assert '[data-testid="stSidebarCollapsedControl"]' in css
    assert '[data-testid="stSidebarCollapseButton"]' in css
    assert '[data-testid="stToolbar"],' not in css
    assert '[data-testid="stAppDeployButton"]' in css


def test_sidebar_toggle_supports_light_theme(monkeypatch) -> None:
    """Sidebar controls have light-mode colors as well as dark-mode colors."""
    fake = _fake_streamlit(monkeypatch, light=True, clicked=False)
    app._styles()
    light_css = fake.markdown.call_args_list[-1].args[0]
    assert '[data-testid="stSidebarCollapsedControl"] button' in light_css
    assert '[data-testid="stSidebarCollapseButton"] button' in light_css
