#!/usr/bin/env python3
import asyncio
import logging
from pymodbus.client import AsyncModbusTcpClient

# enable debug logging if you like
logging.basicConfig()
log = logging.getLogger()
log.setLevel(logging.INFO)

POLL_INTERVAL = 1.0  # seconds between reads

async def main():
    # 1) Create TCP client
    client = AsyncModbusTcpClient("192.168.5.120", port=6502, timeout=3)
    await client.connect()
    if not client.connected:
        print("Failed to connect")
        return

    try:
        while True:
            rr = await client.read_input_registers(address=110, count=16, slave=1)
            if rr.isError():
                print(f"[{asyncio.get_event_loop().time():.0f}] Read error: {rr}")
            else:
                print(f"[{asyncio.get_event_loop().time():.0f}] Registers:", rr.registers)

            # 3) wait before next poll
            await asyncio.sleep(POLL_INTERVAL)
    except asyncio.CancelledError:
        # allow graceful shutdown if cancelled
        pass
    finally:
        # 4) Close connection synchronously
        client.close()

if __name__ == "__main__":
    # Run until stopped (e.g. Ctrl+C)
    try:
        asyncio.run(main(), debug=False)
    except KeyboardInterrupt:
        print("Polling stopped by user.")
