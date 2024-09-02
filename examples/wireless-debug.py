import subprocess
import qrcode
import time

from qrcode.main import QRCode


def generate_qrcode(ip, port):
    """生成二维码

    Args:
        ip: 设备 IP 地址
        port: 设备端口
    """
    data = f"tcp:{ip}:{port}"
    qr = QRCode(
        version=3,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,

    )
    qr.add_data(data)
    qr.print_ascii()

def start_adb_server(port=5555):
    """启动 ADB TCPIP 服务

    Args:
        port: 监听端口
    """
    subprocess.call(["adb", "tcpip", str(port)])


def connect_device(ip, port):
    """连接设备

    Args:
        ip: 设备 IP 地址
        port: 设备端口
    """
    subprocess.call(["adb", "connect", f"{ip}:{port}"])


if __name__ == "__main__":
    # 设备 IP 和端口号
    device_ip = "192.168.10.200"
    device_port = 5555

    # 启动 ADB TCPIP 服务
    start_adb_server(device_port)

    # 生成二维码
    generate_qrcode(device_ip, device_port)

    print("二维码已生成，请在设备上扫描。")
    print("连接成功后，您可以使用 adb 命令进行调试。")
