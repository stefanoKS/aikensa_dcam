
#!/usr/bin/env python3
import asyncio
import logging
from PyQt5.QtCore import QThread, pyqtSignal
from pymodbus.client import AsyncModbusTcpClient

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

class ModbusClientThread(QThread):
    """
    A QThread that runs an asyncio Modbus TCP client in its own event loop,
    polls both Input and Holding Registers, and emits signals when values change.
    """
    holdingUpdated = pyqtSignal(dict)
    inputRead      = pyqtSignal(dict)
    robotConnectionSignal = pyqtSignal(bool)

    def __init__(self,
                 host: str,
                 port: int = 502,
                 slave_id: int = 1,
                 start_addr: int = 0,
                 count: int = 100,
                 poll_interval: float = 0.1,
                 parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.slave_id = slave_id
        self.start_addr = start_addr
        self.count = count
        self.poll_interval = poll_interval

        self._loop = None
        self._client = None

    async def _setup(self):
        """Async setup: create client, connect, and start polling."""
        self._client = AsyncModbusTcpClient(self.host, port=self.port)
        connected = await self._client.connect()
        if not connected:
            _logger.error(f"Failed to connect to {self.host}:{self.port}")
            self.robotConnectionSignal.emit(False)
            return
        _logger.info(f"Connected to {self.host}:{self.port}")
        self._loop.create_task(self._poll_loop())
        

    async def _poll_loop(self):
        prev_input = None
        prev_holding = None

        while True:
            # Read Holding Registers (FC=3)
            try:
                hr = await self._client.read_holding_registers(
                    address=self.start_addr,
                    count=self.count,
                    slave=self.slave_id
                )
                holding = hr.registers if not hr.isError() else None
            except Exception as e:
                _logger.error(f"Error reading Holding Registers: {e}")
                holding = None

            # Read Input Registers (FC=4)
            try:
                ir = await self._client.read_input_registers(
                    address=self.start_addr,
                    count=self.count,
                    slave=self.slave_id
                )
                inputs = ir.registers if not ir.isError() else None
            except Exception as e:
                _logger.error(f"Error reading Input Registers: {e}")
                inputs = None

            # Emit Holding if changed
            if holding is not None:
                data_h = {self.start_addr + i: holding[i] for i in range(len(holding))}
                if data_h != prev_holding:
                    prev_holding = data_h
                    self.holdingUpdated.emit(data_h)

            # Emit Input if changed
            if inputs is not None:
                data_i = {self.start_addr + i: inputs[i] for i in range(len(inputs))}
                if data_i != prev_input:
                    prev_input = data_i
                    self.inputRead.emit(data_i)

            await asyncio.sleep(self.poll_interval)
            #Send signal to show that the robot is connected
            self.robotConnectionSignal.emit(True)


    def run(self):
        """QThread entry point: start and run the asyncio loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        # Schedule async setup under the running loop
        self._loop.create_task(self._setup())

        try:
            self._loop.run_forever()
        finally:
            if self._client:
                self._client.close()
            self._loop.close()

    def stop(self):
        """Stop polling and shut down the event loop."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def write_holding_registers(self, start_addr: int, values: list[int]):
        """Thread-safe write to Holding Registers (FC=16)."""
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

        asyncio.run_coroutine_threadsafe(_do_write(), self._loop)
