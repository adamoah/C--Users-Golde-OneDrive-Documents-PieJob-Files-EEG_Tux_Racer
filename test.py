from pynput import keyboard
from pynput.keyboard import Key, Controller
def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))

def on_release(key):
    print('{0} released'.format(
        key))
    if key == Key.esc:
        # Stop listener
        keyboard = Controller()
        keyboard.type('yella breez')
        return False


while True:
    with keyboard.Listener(on_press=on_press,on_release=on_release) as listener:
        listener.join()