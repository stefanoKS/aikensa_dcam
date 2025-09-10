import pygame

pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")
picking_sound = pygame.mixer.Sound("aikensa/sound/mixkit-kids-cartoon-close-bells-2256.wav")
picking_sound_v2 = pygame.mixer.Sound("aikensa/sound/mixkit-page-forward-single-chime-1107.wav")
keisoku_sound = pygame.mixer.Sound("aikensa/sound/mixkit-bell-notification-933.wav") 
konpou_sound = pygame.mixer.Sound("aikensa/sound/mixkit-software-interface-back-2575.wav")
ok_sound = pygame.mixer.Sound("aikensa/sound/positive_interface.wav")
ng_sound = pygame.mixer.Sound("aikensa/sound/mixkit-classic-short-alarm-993.wav")
announce_sound = pygame.mixer.Sound("aikensa/sound/announce.wav")
announce_sound_2 = pygame.mixer.Sound("aikensa/sound/announce_2.wav")

def play_ok_sound():
    ok_sound.play()

def play_ng_sound():
    ng_sound.play()

def play_alarm_sound():
    alarm_sound.play() 

def play_picking_sound():
    picking_sound_v2.play()

def play_keisoku_sound():
    keisoku_sound.play()

def play_konpou_sound():
    konpou_sound.play()

def play_announce_sound():
    announce_sound.play()   

def play_announce_sound_2():
    announce_sound_2.play()
