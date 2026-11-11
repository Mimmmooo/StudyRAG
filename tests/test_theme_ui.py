"""Regression tests for the theme switch and rendered color schemes."""

from unittest.mock import Mock

import app


def _fake_streamlit(monkeypatch, *, light: bool, clicked: bool) -> Mock:
    """Create an in-memory Streamlit interface without opening a browser."""
    fake = Mock()
    fake.session_state = {"light_mode": light}
    fake.button.return_value = clicked
    monkeypatch.setattr(app, "st", fake)
    return fake


def test_theme_button_updates_state_and_reruns(monkeypatch) -> None:
    """The button must actually switch themes instead of only changing its icon."""
    fake = _fake_streamlit(monkeypatch, light=False, clicked=True)
    app._theme_switch()
    assert fake.session_state["light_mode"] is True
    fake.rerun.assert_called_once()
    assert fake.button.call_args.kwargs["key"] == "theme_button"
    assert "Light" in fake.button.call_args.args[0]


def test_theme_switch_can_return_to_dark(monkeypatch) -> None:
    """A second click restores the dark theme and reruns the UI."""
    fake = _fake_streamlit(monkeypatch, light=True, clicked=True)
    app._theme_switch()
    assert fake.session_state["light_mode"] is False
    fake.rerun.assert_called_once()
    assert "Dark" in fake.button.call_args.args[0]


def test_theme_button_idle_does_not_rerun(monkeypatch) -> None:
    """Ordinary chat rerenders should preserve the current mode."""
    fake = _fake_streamlit(monkeypatch, light=True, clicked=False)
    app._theme_switch()
    assert fake.session_state["light_mode"] is True
    fake.rerun.assert_not_called()


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
