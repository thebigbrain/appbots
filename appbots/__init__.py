import subprocess


def launchApp(name: str):
    try:
        subprocess.run(["adb", "shell", "am", "start", "-n", name])
    except RuntimeError:
        print("Error launching app")
