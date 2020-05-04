# https://kknews.cc/zh-tw/code/rygylen.html

import sys, math, pygame
from datetime import datetime, date, time


def print_text(font, x, y, text, color):
    img_text = font.render(text, True, color)
    screen.blit(img_text, (x, y))


def wrap_angle(angle):
    return abs(angle % 360)


pygame.init()
screen = pygame.display.set_mode((600, 500))
pygame.display.set_caption("Analog Clock Demo")
font = pygame.font.Font(None, 24)
orange = (200, 180, 0)
white = (255, 255, 255)
yellow = (255, 255, 0)
pink = (255, 100, 100)

pos_x = 300
pos_y = 250
radius = 250
angle = 360


def draw_hand(cur_time, time_unit, hand_length, color, width):
    # [1] angle % 360
    time_angle = wrap_angle(cur_time * (360 / time_unit) - 90)
    # [2] convert angle from degrees to radians
    time_angle = math.radians(time_angle)
    x_offset = math.cos(time_angle) * (radius - hand_length)
    y_offset = math.sin(time_angle) * (radius - hand_length)
    target = (pos_x + x_offset, pos_y + y_offset)
    pygame.draw.line(screen, color, (pos_x, pos_y), target, width)


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
    keys = pygame.key.get_pressed()
    if keys[pygame.K_ESCAPE]:
        screen.fill(0, 0, 100)
        sys.exit()

    # draw the clock numbers 1-12
    for n in range(1, 13):
        angle = math.radians(n * (360 / 12) - 90)
        x = math.cos(angle) * (radius - 20) - 10
        y = math.sin(angle) * (radius - 20) - 10
        print_text(font, pos_x + x, pos_y + y, str(n), yellow)

    # get the time of day
    today = datetime.today()
    hours = today.hour
    minutes = today.minute
    seconds = today.second

    # draw the hours hand
    draw_hand(hours, time_unit=12, hand_length=80, color=pink, width=25)

    # draw the minute hand
    draw_hand(minutes, time_unit=60, hand_length=60, color=orange, width=12)

    # draw the second hand
    draw_hand(seconds, time_unit=60, hand_length=40, color=yellow, width=6)

    # convert the center
    pygame.draw.circle(screen, white, (pos_x, pos_y), radius, 6)
    pygame.draw.circle(screen, white, (pos_x, pos_y), 20)
    print_text(font, 0, 0, str(hours) + " : " + str(minutes) + " : " + str(seconds), pink)

    pygame.display.update()
    screen.fill((0, 0, 0))



