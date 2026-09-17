"""Tests for the contextvars-carrying memory-stream wrappers."""

import anyio
import pytest

from mcp.shared._context_streams import create_context_streams

pytestmark = pytest.mark.anyio


async def test_sync_close_closes_the_underlying_streams() -> None:
    """The wrappers mirror anyio's memory streams: close() is the sync form of aclose()."""
    send, receive = create_context_streams[str](1)
    await send.send("queued")
    send.close()
    receive.close()
    with pytest.raises(anyio.ClosedResourceError):
        await send.send("after close")
    with pytest.raises(anyio.ClosedResourceError):
        await receive.receive()


def test_create_context_streams_marks_the_receive_side_as_a_half_close_on_request() -> None:
    """The receive side carries the transport's EOF semantics; a full close is the default."""
    plain_send, plain_recv = create_context_streams[int](0)
    half_close_send, half_close_recv = create_context_streams[int](0, eof_is_half_close=True)
    try:
        assert plain_recv.eof_is_half_close is False
        assert half_close_recv.eof_is_half_close is True
    finally:
        for s in (plain_send, plain_recv, half_close_send, half_close_recv):
            s.close()
