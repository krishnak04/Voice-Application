import asyncio
import aiohttp

async def test():
    urls = ['ws://localhost:8024/ws/call', 'ws://127.0.0.1:8024/ws/call']
    origins = ['http://localhost:8023', 'http://127.0.0.1:8023', 'http://example.com', '']
    for url in urls:
        for origin in origins:
            headers = {'Origin': origin} if origin else {}
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as ws:
                        msg = await ws.receive_json(timeout=5)
                        print(f"[PASS] {url} (Origin: '{origin}'): received {msg.get('type')}")
            except Exception as e:
                print(f"[FAIL] {url} (Origin: '{origin}'): {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test())
