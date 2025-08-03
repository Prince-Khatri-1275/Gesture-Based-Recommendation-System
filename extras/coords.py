import mouse
import pyautogui
import keyboard
import time

while True:
    if keyboard.is_pressed("q"):
        break

    if mouse.is_pressed(mouse.LEFT):
        print(mouse.get_position())
        """ LEFT
            (663, 477)
            (717, 475)
            (691, 454)
            (689, 509)
            
            700, 550 not 480
        """
        """ RIGHT
            (764, 475)
            (788, 452)
            (814, 481)
            (791, 508)
            
            800, 550 # not 480  
        """
        time.sleep(0.2)

print("Released")