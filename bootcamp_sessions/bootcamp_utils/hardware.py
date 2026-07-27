"""
hardware.py — shared Grove peripheral helpers (LED bar and OLED screen).

Several notebooks plug small "Grove" gadgets into the PYNQ Grove Adapter:

  - The Grove LED Bar (10 LEDs in a row) — used in PYNQ 101 and 201.
  - The Grove OLED screen (a tiny text display) — used in PYNQ 301.

Setting these up is the same every time, so the setup lives here once and is
shared by every notebook that uses them.
"""
from pynq_peripherals import PmodGroveAdapter


def setup_led_bar(overlay, port="G2"):
    """
    Set up the Grove LED Bar and start with all LEDs off.

    Parameters
    ----------
    overlay : the overlay from load_dpu() (or a DpuOverlay).
    port : which Grove port the LED bar is plugged into (default "G2").

    Returns
    -------
    ledbar : the LED bar object. Pass it to set_led_bar() to light LEDs.

    Example
    -------
        ledbar = setup_led_bar(overlay)      # LED bar plugged into G2
    """
    adapter = PmodGroveAdapter(overlay.PMODA, **{port: "grove_ledbar"})
    ledbar = getattr(adapter, port)
    ledbar.clear()   # start fresh with all LEDs off
    return ledbar


def set_led_bar(ledbar, level, brightness=3, green_to_red=1):
    """
    Light up the LED bar to a given level (0-10).

    Parameters
    ----------
    ledbar : the LED bar from setup_led_bar().
    level : how many LEDs to light, 0 (off) to 10 (all on).
    brightness : how bright, 0 to 255 (default 3).
    green_to_red : color gradient direction (default 1).

    Example
    -------
        set_led_bar(ledbar, 7)     # light up 7 of the 10 LEDs
        set_led_bar(ledbar, 0)     # turn them all off
    """
    ledbar.clear()
    ledbar.set_level(int(level), brightness, green_to_red)


def setup_oled(overlay, port="G4"):
    """
    Set up the Grove OLED text screen.

    Parameters
    ----------
    overlay : the overlay from load_dpu() (or a DpuOverlay).
    port : which Grove port the OLED is plugged into (default "G4").

    Returns
    -------
    oled : the OLED object. Pass it to show_on_oled() to display text.

    Example
    -------
        oled = setup_oled(overlay)     # OLED plugged into G4
    """
    adapter = PmodGroveAdapter(overlay.PMODA, **{port: "grove_oled"})
    oled = getattr(adapter, port)
    oled.set_default_config()
    oled.set_normal_display()
    return oled


def show_on_oled(oled, text):
    """
    Clear the OLED screen and show a line of text.

    Example
    -------
        show_on_oled(oled, "cat")
    """
    oled.clear_display()
    oled.put_string(str(text))
