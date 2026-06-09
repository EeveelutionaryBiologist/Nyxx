
import datetime
import mss

from PIL import Image

"""
Provided functionality is currently broken on many OS/Compositors 
due to security guidelines.
Maybe we will find a better solution down the line.
"""

def universal_screen_capture_fallback(filename: Path):
    """
    An aggressive OS-level fallback strategy. Checks what native screen-capture
    binaries are installed on the user's Linux system and leverages them.
    """
    # 1. Check for grim (The definitive standard for Wayland/Screencopy protocols)
    if shutil.which("grim"):
        subprocess.run(["grim", str(filename)], check=True)
        return True
        
    # 2. Check for spectacle (KDE Plasma native CLI engine)
    if shutil.which("spectacle"):
        subprocess.run(["spectacle", "-b", "-n", "-o", str(filename)], check=True)
        return True

    # 3. Check for gnome-screenshot (GNOME desktop native engine)
    if shutil.which("gnome-screenshot"):
        subprocess.run(["gnome-screenshot", "-f", str(filename)], check=True)
        return True

    # 4. Check for scrot (The classic lightweight X11 fallback tool)
    if shutil.which("scrot"):
        subprocess.run(["scrot", str(filename)], check=True)
        return True

    return False


def capture_full_screen_mss():
    """
    Captures a screenshot across X11 and Wayland compositors
    using the mss library, exporting a standard PIL Image instance.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = TMP_DIR / f"fullscreen_{timestamp}.png"

    with mss.mss() as sct:
        # sct.monitors[0] captures the absolute unified geometry (all monitors stitched)
        # sct.monitors[1] is monitor 1, sct.monitors[2] is monitor 2, etc.
        primary_monitor = sct.monitors[0]
        
        # Capture raw pixels from screen memory buffer
        sct_img = sct.grab(primary_monitor)
        
        # Convert the raw mss pixel structure into a standard Pillow Image object
        screenshot = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # Persist to disk for your agent's vision tool pipeline
        screenshot.save(filename, "PNG")
        
    print(f"[SYSTEM] Screenshot captured -> {filename}")
    return screenshot, filename
