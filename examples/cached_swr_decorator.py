import asyncio

from aiocache import cached_swr, SimpleMemoryCache


class FakeClock:
    """Controllable time source so expiration can be tested without sleeping."""

    def __init__(self, start=1000.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


clock = FakeClock()
cache = SimpleMemoryCache()
calls = 0


@cached_swr(cache, soft_ttl=10, hard_ttl=20, time_func=clock)
async def hot_config():
    global calls
    calls += 1
    return {"version": calls}


async def test_cached_swr():
    async with cache:
        # Fresh phase: first call computes, following calls reuse the fresh value.
        assert (await hot_config()) == {"version": 1}
        assert (await hot_config()) == {"version": 1}
        assert calls == 1

        # Stale phase: soft_ttl elapsed, stale value is served instantly while a
        # single background refresh runs per cache instance and key.
        clock.advance(15)
        assert (await hot_config()) == {"version": 1}
        assert (await hot_config()) == {"version": 1}
        while calls == 1:  # wait for the single background refresh to finish
            await asyncio.sleep(0)
        assert calls == 2
        assert (await hot_config()) == {"version": 2}

        # Hard-expired phase: past hard_ttl callers must wait for a fresh result.
        clock.advance(25)
        assert (await hot_config()) == {"version": 3}
        assert calls == 3

    await hot_config.close()


if __name__ == "__main__":
    asyncio.run(test_cached_swr())
