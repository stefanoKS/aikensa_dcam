import asyncio
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from pymodbus.client import AsyncModbusTcpClient

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ModbusClientThread(QThread):
    """
    A QThread that runs an asyncio Modbus TCP client,
    polls the PLC for its Input Registers and Holding Registers,
    and emits signals whenever they change. It also allows
    thread-safe writes to Holding Registers on the PLC.
    """
    # Signals to mirror your old server-side names
    holdingUpdated = pyqtSignal(dict)
    inputRead      = pyqtSignal(dict)

    def __init__(self,
                 host: str,
                 port: int = 502,
                 slave_id: int = 1,
                 start_addr: int = 0,
                 count: int = 100,
                 poll_interval: float = 0.5,
                 parent=None):
        super().__init__(parent)
        self.host          = host
        self.port          = port
        self.slave_id      = slave_id
        self.start_addr    = start_addr
        self.count         = count
        self.poll_interval = poll_interval

        self._client = None
        self._loop   = None

    async def _poll_input_loop(self):
        prev = None
        while True:
            await asyncio.sleep(self.poll_interval)
            if not self._client or not self._client.connected:
                continue
            try:
                rr = await self._client.read_input_registers(
                    address=self.start_addr,
                    count=self.count,
                    slave=self.slave_id
                )
                raw = rr.registers
            except Exception as e:
                _logger.error(f"Error reading Input Registers: {e}")
                continue

            data = {self.start_addr + i: raw[i] for i in range(len(raw))}
            if data != prev:
                prev = data
                self.inputRead.emit(data)

    async def _poll_holding_loop(self):
        prev = None
        while True:
            await asyncio.sleep(self.poll_interval)
            if not self._client or not self._client.connected:
                continue
            try:
                rr = await self._client.read_holding_registers(
                    address=self.start_addr,
                    count=self.count,
                    slave=self.slave_id
                )
                raw = rr.registers
            except Exception as e:
                _logger.error(f"Error reading Holding Registers: {e}")
                continue

            data = {self.start_addr + i: raw[i] for i in range(len(raw))}
            if data != prev:
                prev = data
                self.holdingUpdated.emit(data)

    async def _run_client(self):
        # Establish connection
        self._client = AsyncModbusTcpClient(self.host, port=self.port)
        await self._client.connect()
        _logger.info(f"Modbus client connected to {self.host}:{self.port}")

        # Schedule both polling loops
        self._loop.create_task(self._poll_input_loop())
        self._loop.create_task(self._poll_holding_loop())

        # Block forever so thread stays alive
        await asyncio.Event().wait()

    def run(self):
        # Set up and run the asyncio loop in this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_client())
        finally:
            self._loop.close()

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def write_holding_registers(self, start_addr: int, values: list[int]):
        """
        Thread‐safe way to write into Holding Registers on the PLC
        using function code 16.
        """
        if not self._loop or not self._client:
            _logger.warning("Cannot write: client not running yet")
            return

        async def _do_write():
            try:
                rr = await self._client.write_registers(
                    address=start_addr,
                    values=values,
                    slave=self.slave_id
                )
                if rr.isError():
                    _logger.error(f"Write HR error: {rr}")
                else:
                    _logger.info(f"Wrote HR[{start_addr}..] = {values}")
            except Exception as e:
                _logger.error(f"Exception writing HR: {e}")

        # Schedule on the event loop
        asyncio.run_coroutine_threadsafe(_do_write(), self._loop)
