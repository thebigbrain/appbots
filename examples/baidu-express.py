from mobly.controllers import android_device
from snippet_uiautomator import uiautomator
from snippet_uiautomator.uiautomator import ToolType
from snippet_uiautomator.uidevice import UiDevice

ads = android_device.create(android_device.ANDROID_DEVICE_PICK_ALL_TOKEN)

ad = ads[0]

configurator = uiautomator.Configurator(
    flags=[
        uiautomator.Flag.FLAG_DONT_SUPPRESS_ACCESSIBILITY_SERVICES,
    ],
    tool_type=ToolType.TOOL_TYPE_FINGER
)
ad.services.register(
    uiautomator.ANDROID_SERVICE_NAME,
    uiautomator.UiAutomatorService,
    uiautomator.UiAutomatorConfigs(configurator=configurator, skip_installing=True),
)

ui: UiDevice = ad.ui

ui.wakeup()
ui.press.home()

ui().swipe.left(percent=50)
print(ui().exists)
