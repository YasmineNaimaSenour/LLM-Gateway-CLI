"""Tests for the shared HTTP retry helper.

The contract under test: retry connection errors and 500/502/503/504 with
exponential backoff; never retry 429, other 4xx, non-conventional 5xx, or
read timeouts; hand the last response back when retries are exhausted so the
caller's classification keeps its final say.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from src.providers.http_utils import post_with_retry


def _resp(status_code: int) -> MagicMock:
    return MagicMock(status_code=status_code)


@patch("src.providers.http_utils.requests.post")
def test_returns_response_immediately_when_ok(mock_post):
    mock_post.return_value = _resp(200)

    resp = post_with_retry("http://x", timeout=1)

    assert resp.status_code == 200
    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# retried: connection errors and 500/502/503/504
# ---------------------------------------------------------------------------


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_connection_error_is_retried_then_raised(mock_post, mock_sleep):
    mock_post.side_effect = requests.exceptions.ConnectionError("refused")

    with pytest.raises(requests.exceptions.ConnectionError):
        post_with_retry("http://x", timeout=1)

    assert mock_post.call_count == 3  # initial attempt + 2 retries


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_connection_error_recovers_on_retry(mock_post, mock_sleep):
    mock_post.side_effect = [
        requests.exceptions.ConnectionError("blip"),
        _resp(200),
    ]

    resp = post_with_retry("http://x", timeout=1)

    assert resp.status_code == 200
    assert mock_post.call_count == 2


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_conventional_5xx_is_retried(mock_post, mock_sleep, status):
    mock_post.return_value = _resp(status)

    resp = post_with_retry("http://x", timeout=1)

    assert resp.status_code == status
    assert mock_post.call_count == 3


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_5xx_recovers_on_retry(mock_post, mock_sleep):
    mock_post.side_effect = [_resp(502), _resp(502), _resp(200)]

    resp = post_with_retry("http://x", timeout=1)

    assert resp.status_code == 200
    assert mock_post.call_count == 3


# ---------------------------------------------------------------------------
# never retried: 429, other 4xx, non-conventional 5xx, read timeouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [429, 400, 401, 403, 404, 422, 501])
@patch("src.providers.http_utils.requests.post")
def test_non_retryable_status_is_returned_without_retry(mock_post, status):
    mock_post.return_value = _resp(status)

    resp = post_with_retry("http://x", timeout=1)

    assert resp.status_code == status
    assert mock_post.call_count == 1


@patch("src.providers.http_utils.requests.post")
def test_read_timeout_is_not_retried(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout("read timed out")

    with pytest.raises(requests.exceptions.Timeout):
        post_with_retry("http://x", timeout=1)

    assert mock_post.call_count == 1


# ---------------------------------------------------------------------------
# backoff and exhaustion semantics
# ---------------------------------------------------------------------------


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_backoff_is_exponential(mock_post, mock_sleep):
    mock_post.return_value = _resp(503)

    post_with_retry("http://x", timeout=1, max_retries=2)

    assert [call.args[0] for call in mock_sleep.call_args_list] == [0.5, 1.0]


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_exhausted_retries_hand_back_the_last_response(mock_post, mock_sleep):
    last = _resp(503)
    mock_post.return_value = last

    resp = post_with_retry("http://x", timeout=1, max_retries=1)

    assert resp is last
    assert mock_post.call_count == 2


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_exhaustion_respects_custom_max_retries(mock_post, mock_sleep):
    mock_post.return_value = _resp(500)

    post_with_retry("http://x", timeout=1, max_retries=4)

    assert mock_post.call_count == 5
    assert mock_sleep.call_count == 4


# ---------------------------------------------------------------------------
# user-facing behavior
# ---------------------------------------------------------------------------


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_retry_prints_notice_to_stderr(mock_post, mock_sleep, capsys):
    mock_post.return_value = _resp(503)

    post_with_retry("http://x", timeout=1)

    err = capsys.readouterr().err
    assert "[http-retry]" in err
    assert "503" in err
    assert "0.5s" in err


@patch("src.providers.http_utils.time.sleep")
@patch("src.providers.http_utils.requests.post")
def test_no_stderr_output_when_first_attempt_succeeds(mock_post, mock_sleep, capsys):
    mock_post.return_value = _resp(200)

    post_with_retry("http://x", timeout=1)

    assert capsys.readouterr().err == ""
