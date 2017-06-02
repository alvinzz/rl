import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle

def draw_x(ax, center):
    center_x = center[0]
    center_y = center[1]
    corners = []
    corners.append((center_x - np.random.uniform(5, 55), center_y - np.random.uniform(5, 55)))
    corners.append((center_x + np.random.uniform(5, 55), center_y - np.random.uniform(5, 55)))
    corners.append((center_x + np.random.uniform(5, 55), center_y + np.random.uniform(5, 55)))
    corners.append((center_x - np.random.uniform(5, 55), center_y + np.random.uniform(5, 55)))
    ax.plot((corners[0][0], corners[2][0]), (corners[0][1], corners[2][1]), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))
    ax.plot((corners[1][0], corners[3][0]), (corners[1][1], corners[3][1]), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))

def draw_o(ax, center):
    radius = np.random.uniform(5, 55)
    center_x = center[0] + np.random.uniform(-(50 - radius * 0.75), 50 - radius * 0.75)
    center_y = center[1] + np.random.uniform(-(50 - radius * 0.75), 50 - radius * 0.75)
    circ = plt.Circle((center_x, center_y), radius, color=str(np.random.uniform(0, 0.5)), fill=False, linewidth=np.random.uniform(1, 3))
    ax.add_artist(circ)

def create_training_set(items=1000):
    for i in range(items):
        fig = plt.figure(figsize=(512 / 96, 512 / 96), dpi=96)
        ax = fig.add_subplot(111)
        ax.plot((-50, -50), (-150, 150), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))
        ax.plot((50, 50), (-150, 150), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))
        ax.plot((-150, 150), (-50, -50), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))
        ax.plot((-150, 150), (50, 50), color=str(np.random.uniform(0, 0.5)), linewidth=np.random.uniform(1, 3))

        turn = np.random.randint(2)
        pixel_centers = [(-100, 100), (0, 100), (100, 100), (-100, 0), (0, 0), (100, 0), (-100, -100), (0, -100), (100, -100)]
        actions = set(range(9))
        state = [0] * 18
        game_len = np.random.randint(10)
        turn_num = 0
        while actions and turn_num < game_len:
            action = np.random.choice(tuple(actions))
            actions.remove(action)
            center = pixel_centers[action]
            turn = 1 - turn
            if turn:
                draw_x(ax, center)
                state[action] = 1
            else:
                draw_o(ax, center)
                state[action + 9] = 1
            turn_num += 1

        background = np.random.randint(128, 256)
        ax.set_axis_bgcolor(str(background / 256))
        ax.set_xlim([-200, 200])
        ax.set_ylim([-200, 200])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        fig.subplots_adjust(bottom=0.,left=0.,right=1.,top=1.)
        plt.savefig('tmp/fig.png', dpi=96)
        
        plt.cla()
        plt.clf()
        plt.close()

        im = cv2.imread('tmp/fig.png', 0)

        pts1 = np.float32([[0,0],[512,0],[512,512],[0,512]])
        pts2 = np.float32([[np.random.uniform(-50, 100), np.random.uniform(-50, 100)],
            [512 - np.random.uniform(-50, 100), np.random.uniform(-50, 100)],
            [512 - np.random.uniform(-50, 100), 512 - np.random.uniform(-50, 100)],
            [np.random.uniform(-50, 100), 512 - np.random.uniform(-50, 100)]])

        M = cv2.getPerspectiveTransform(pts1, pts2)
        im = cv2.warpPerspective(im, M, (512, 512), borderValue=background)

        cv2.imwrite('images/{}.png'.format(i), im)
        pickle.dump(state, open("labels/{}.p".format(i), "wb"))

if __name__ == "__main__":
    create_training_set()
