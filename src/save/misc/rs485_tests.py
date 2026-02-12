import serial
import time

# connect A to B, leave GND hanging, to test cable + adapter
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
time.sleep(1)

# Send some bytes
test_data = b'\xCC\x07\x00'
print(f"Sending: {test_data.hex()}")
ser.write(test_data)

# Read back (should be same bytes if A/B shorted)
time.sleep(0.1)
received = ser.read(10)
print(f"Received: {received.hex()}")

if received == test_data:
    print("✅ RS485 adapter working - loopback successful")
else:
    print("❌ RS485 adapter problem - no loopback")

ser.close()

#
# ## test cable alone
# import serial
# import time
#
# ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
#
# # Send some data (even though nothing is connected to receive it)
# ser.write(b'Hello World')
# print("Data sent - if no error, FTDI cable is working")
#
# # Try to read (should timeout since nothing connected)
# data = ser.read(10)
# print(f"Received: {data}")  # Should be empty
#
# ser.close()