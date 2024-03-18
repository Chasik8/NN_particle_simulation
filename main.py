import math
import random
import time

import pygame as pg
import pymunk.pygame_util
import torch
from tqdm import tqdm
import numpy as np
from model import *


def Train_init(kl_kol):
    k = 2
    btach = 500
    file_name = f"train\{str(k)}.npy"
    data = np.load(file_name, allow_pickle=True)
    x_train_inp = data.item().get('a')
    y_train_inp = data.item().get('b')
    time_line = data.item().get('time')
    w = data.item().get('w')
    h = data.item().get('h')
    kl = min(w, h) // kl_kol
    net_k_out = math.ceil(w / kl) * math.ceil(h / kl)
    net_k_inp = len(x_train_inp) * 2 + 1
    x_train = []
    for l in range(len(y_train_inp)):
        dop = []
        for j in x_train_inp.tolist():
            dop.append(j[0])
            dop.append(j[1])
        dop.append(time_line[l])
        x_train.append(dop)
    y_train = []
    for l in range(len(y_train_inp)):
        y_train.append([])
        for i in range(math.ceil(w / kl)):
            y_train[l].append([])
            for j in range(math.ceil(h / kl)):
                y_train[l][i].append(0)
    for l in range(len(y_train_inp)):
        for i in y_train_inp[l]:
            # print(int(i[0] // kl))
            y_train[l][int(i[0] // kl)][int(i[1] // kl)] = 1
        y_train[l] = np.array(y_train[l]).reshape(-1)
    # x_train = np.expand_dims(x_train, axis=1)
    x_train = torch.from_numpy(np.array(x_train).astype(np.float32))
    y_train = torch.from_numpy(np.array(y_train).astype(np.float32))
    return x_train, y_train, net_k_inp, net_k_out


def Run():
    k_model = 0
    train_dop = False
    try:
        ff = open('conf_model.txt', 'r')
        k_model = int(ff.read())
        ff.close()
        ff = open('conf_model.txt', 'w')
        ff.write(str(k_model + 1))
        ff.close()
    except:
        ff = open('conf_model.txt', 'w')
        ff.write(str(1))
        ff.close()
    dev = torch.device("cuda:0")
    splitting = 400
    x_train, y_train, net_k_inp, net_k_out = Train_init(splitting)
    x_train = x_train.to(device=dev)
    y_train = y_train.to(device=dev)
    net = Net(net_k_inp, net_k_out)
    if train_dop:
        PATH = f"models\model{str(k_model - 1)}.pth"
        net.load_state_dict(torch.load(PATH))
        net.eval()
    net.to(dev)
    criterion = nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(net.parameters())
    epoch_tim = time.time()
    print("Run")
    loss_max = 1000000000000000000000000
    if train_dop:
        ft = open(f"floss_dir\\floss_max.txt", 'r')
        loss_max = float(ft.read())
        ft.close()
    for epoch in range(net.num_epochs):
        # for i, (images, labels) in enumerate(train_loader):  # Загрузка партии изображений с индексом, данными,
        # классом
        loss = 0
        sr_loss = 0
        for i in tqdm(range(len(x_train))):
            optimizer.zero_grad()  # Инициализация скрытых масс до нулей
            outputs = net(x_train[i])  # Передний пропуск: определение выходного класса, данного изображения
            loss = criterion(outputs, y_train[i])  # Определение потерь: разница между выходным классом и предварительно
            # заданной
            # меткой
            loss.backward()
            # torch.cuda.empty_cache()  # Обратный проход: определение параметра weight
            optimizer.step()
            sr_loss += loss.item()
        print(epoch, sr_loss / len(x_train))
        if sr_loss / len(x_train) < loss_max:
            loss_max = sr_loss / len(x_train)
            torch.save(net.state_dict(), fr"models\model{k_model}_max.pth")
            floss_max = open("floss_dir\\floss_max.txt", 'w')
            floss_max.write(str(sr_loss / len(x_train)))
            floss_max.close()
        if epoch % 10 == 0:
            torch.save(net.state_dict(), fr"models\model{k_model}.pth")
    torch.save(net.state_dict(), fr"models\model{k_model}.pth")
    print(time.time() - epoch_tim, (time.time() - epoch_tim) / net.num_epochs)


def create_image_rand(place, pos, r):
    body = pymunk.Body(1, 100, body_type=pymunk.Body.DYNAMIC)
    body.position = pos
    shape = pymunk.Circle(body, r)
    shape.elasticity = 1
    # shape.collision_type = 1
    v = random.uniform(-1, 1)
    if random.randint(0, 1):
        shape.body.velocity = (v, math.sqrt(1 - v ** 2))
    else:
        shape.body.velocity = (v, -math.sqrt(1 - v ** 2))
    place.add(body, shape)
    return shape


def draw_image(images, gameDisplay):
    for image in images:
        pos_x = int(image.body.position.x)
        pos_y = int(image.body.position.y)
        r = int(image.radius)
        pg.draw.circle(gameDisplay, (0, 0, 0), (pos_x, pos_y), r)


def static_floor(place, pos):
    body = pymunk.Body(body_type=pymunk.Body.STATIC)
    body.position = pos
    shape = pymunk.Circle(body, 60)
    place.add(body, shape)
    return shape


def draw_static(floors, gameDisplay, r):
    for floor in floors:
        pos_x = int(floor.body.position.x)
        pos_y = int(floor.body.position.y)
        pg.draw.circle(gameDisplay, (0, 0, 0), (pos_x, pos_y), 40)


def Game(iter, batch):
    place = pymunk.Space()
    Draw = True
    # place.gravity = (0, 500)
    WIDTH = 800
    HEIGHT = 800
    kol = 12
    print(math.sqrt(WIDTH * HEIGHT / kol / math.pi))
    # R = math.sqrt(WIDTH * HEIGHT / kol / math.pi)
    kl = min(WIDTH, HEIGHT) // kol
    R = kl // 2 - 1
    print(R, kl)
    # --------------------------------------------------------------------------------
    walls = [pymunk.Segment(place.static_body, (0, HEIGHT), (WIDTH, HEIGHT), 1),
             pymunk.Segment(place.static_body, (0, 0), (0, HEIGHT), 1),
             pymunk.Segment(place.static_body, (WIDTH, 0), (WIDTH, HEIGHT), 1),
             pymunk.Segment(place.static_body, (0, 0), (WIDTH, 0), 1)
             ]
    for wall in walls:
        wall.elasticity = 1.0  # Make the walls perfectly bouncy
        # wall.collision_type = 1  # Define a collision type for the walls
        place.add(wall)
    # --------------------------------------------------------------------------------
    run = True
    images = []
    # floors = []
    # floors.append(static_floor(place, (400, 400)))
    tr = []
    for i in range(1, WIDTH // kl - 1):
        for j in range(1, HEIGHT // kl - 1):
            if not random.randint(0, 3):
                # images.append(create_image_rand(place, ((i % k) * (2 * R) + R, (i // k) * (2 * R) + R), R))
                images.append(create_image_rand(place, (i * kl, j * kl // 2), R))
                # print(images[-1].body.position[0])
                tr.append([images[-1].body.position.x, images[-1].body.position.y, images[-1].body.velocity.x,
                           images[-1].body.velocity.y])
    k = WIDTH / (2 * R)
    # for i in range(kol):
    #     if not random.randint(0, 20):
    #         # images.append(create_image_rand(place, ((i % k) * (2 * R) + R, (i // k) * (2 * R) + R), R))
    #         images.append(create_image_rand(place, ((i % k) * (2 * R) + R, (i // k) * (2 * R) + R), R))
    #         # print(images[-1].body.position[0])
    #         tr.append([images[-1].body.position.x, images[-1].body.position.y, images[-1].body.velocity.x,
    #                    images[-1].body.velocity.y])
    train_x = tr
    time_line = set()
    while len(time_line) < batch:
        time_line.add(random.randint(1, batch * 100))
    time_line = list(time_line)
    time_line.sort()
    pp = 0
    ppi = 0
    if Draw:
        pg.init()
        gameDisplay = pg.display.set_mode((WIDTH, HEIGHT))
        time = pg.time.Clock()
    train_y = []
    while run:
        if Draw:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()

            gameDisplay.fill((217, 217, 217))
            draw_image(images, gameDisplay)
            # draw_static(floors, gameDisplay)
            place.step(10)
            pg.display.update()
            # time.tick(70)
        place.step(10)
        if pp == time_line[ppi]:
            ppi += 1
            tr = []
            for i in range(len(images)):
                if 0 <= images[i].body.position.x < WIDTH or 0 <= images[i].body.position.y < HEIGHT:
                    tr.append([images[i].body.position.x, images[i].body.position.y, images[i].body.velocity.x,
                               images[i].body.velocity.y])
                else:
                    print("Error")
                    # exit()
            train_y.append(tr)
            # print(ppi, pp)
        pp += 1
        if ppi >= len(time_line):
            break
    np.save(f"train\\{iter}", {'a': np.array(train_x).astype(np.float32),
                               'b': np.array(train_y).astype(np.float32),
                               'time': np.array(time_line).astype(np.float32),
                               "w": WIDTH, 'h': HEIGHT})


def print_hi(name):
    # Game(2, 10 ** 3)
    Run()


if __name__ == '__main__':
    print_hi('PyCharm')
