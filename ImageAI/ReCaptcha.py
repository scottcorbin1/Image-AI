import os.path
import random
import numpy as np
import pygame
import tkinter
from tensorflow import keras
from keras.utils import np_utils
from keras.datasets import cifar10
import matplotlib.pyplot as plt


root = tkinter.Tk()
width = root.winfo_screenwidth()/1.3
height = root.winfo_screenheight()/1.3
fps = 60
gray = (80, 80, 80)
white = (255, 255, 255)
image_width = 100
image_height = 100
button_width = image_width + 2
button_height = image_height + 2
clock = pygame.time.Clock()
show_image = True
running = True
show_right = False
images = [i for i in range(2, 10)]
win = pygame.display.set_mode((width, height))
user_selection = []
model = keras.models.load_model('AI_Trained_Epoch100.h5', compile=False)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
pygame.init()
pygame.display.set_caption('ReCaptcha')


(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize the inputs from 0-255 to between 0 and 1 by dividing by 255
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# One-hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
class_num = y_test.shape[1]
plt.axis('off')

path = "images"
if not os.path.exists("images"):
    os.makedirs(path)

class Button:
    def __init__(self, color, x, y, x_width, y_height):
        self.color = color
        self.x = x
        self.y = y
        self.x_width = x_width
        self.y_height = y_height

    def draw(self, window, outline=None):
        # Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(window, outline, (self.x - 2, self.y - 2, self.x_width + 4, self.y_height + 4), 0)

        pygame.draw.rect(win, self.color, (self.x, self.y, self.x_width, self.y_height), 0)

    def is_over(self, pos):
        # Pos is the mouse position or a tuple of (x,y) coordinates
        if self.x < pos[0] < self.x + self.x_width:
            if self.y < pos[1] < self.y + self.y_height:
                return True

        return False


def ai_master_function(index_list: list, test=False):
    # First Build Prediction Query Data by combining images at index in the list
    data = np.array([X_test[i] for i in index_list])
    # Get Predictions
    raw_predictions = np.array([model.predict(data)])[0]

    # Now Interpret the Data
    # First Build Dictionary for easy reference/Interpret
    def interpreted_prediction(class_id=-1, prob=0):
        return {"class_id": class_id, "probability": (prob * 100)};

    interpreted_predictions = []
    for predictions in raw_predictions:
        pred = [interpreted_prediction()]
        for i, prediction in enumerate(predictions):
            p_test = interpreted_prediction(prob=prediction)["probability"]
            if pred[0]["probability"] > p_test:
                continue
            if pred[0]["probability"] == p_test:
                pred.append(interpreted_prediction(i, prediction))
            if pred[0]["probability"] < p_test:
                pred = [interpreted_prediction(i, prediction)]
        interpreted_predictions.append(pred)

    # Now get the simplified list of results
    predictions = [i[0]['class_id'] for i in interpreted_predictions]
    return predictions


# Gets the random indexes of a number of images from a given data set:
def get_image_indexes(num_of_images, data_set):
    image_list = []

    # Creates a loop that iterates  equal to the number of images given:
    for n in range(num_of_images):

        image_choice = random.randint(0, len(data_set) - 1)

        for image in image_list:
            # Selects an index from 0 to the maximum size of the given dataset:
            if image == image_choice:
                image_choice = random.randint(0, len(data_set) - 1)

        image_list.append(image_choice)

    return image_list


# Picks the Category that will need to be selected by the user:
def pick_category(category_list):

    # Gets the larges index in the given list:
    max_index = len(category_list)-1

    # Selects a random index from the list of possible categories:
    rand_index = random.randint(0, max_index)

    # Gets the category from the list of categories:
    chosen_category = category_list[rand_index]

    return chosen_category


# Returns a list of 1's and 0's with the ones representing the corresponding
# images from the list of images:
def correct_answer_location(ai_chosen_categories, chosen_category ):

    correct_ans_locations = []

    for c in ai_chosen_categories:

        if c == chosen_category:
            correct_ans_locations.append(1)

        else:
            correct_ans_locations.append(0)

    return correct_ans_locations


# Return what percentage of the number of total images where chosen correctly
# by the user:
def verify_choices(user_selections, chosen_category, ai_chosen_categories):
    # Spoof Proofing
    if len(user_selections) == 0:
        return 0
    # Collector
    correct = 0
    for i, possible in enumerate(ai_chosen_categories):
        if possible == chosen_category and i in user_selections:
            correct += 1
        if possible != chosen_category and i not in user_selections:
            correct += 1

    print(correct)
    return round((correct / len(ai_chosen_categories)) * 100, 2)


# Take in a category number and returns what category it is as a string
def get_category(cat=0):
    category = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }
    if cat not in category.keys():
        return "Unknown"
    return category[cat]


def draw_images():
    image1_button.draw(win, (0, 0, 0))
    win.blit(image1, (first_img_width_pos + 1, first_img_height_pos + 1))
    image2_button.draw(win, (0, 0, 0))
    win.blit(image2, (first_img_width_pos + image_width + 20, first_img_height_pos + 1))
    image3_button.draw((win, (0, 0, 0)))
    win.blit(image3, (first_img_width_pos + image_width * 2 + 41, first_img_height_pos + 2))
    image4_button.draw(win, (0, 0, 0))
    win.blit(image4, (first_img_width_pos + 1, first_img_height_pos + image_height + 20))
    image5_button.draw(win, (0, 0, 0))
    win.blit(image5, (first_img_width_pos + image_width + 20, first_img_height_pos + image_height + 20))
    image6_button.draw(win, (0, 0, 0))
    win.blit(image6, (first_img_width_pos + image_width * 2 + 40, first_img_height_pos + image_height + 20))
    image7_button.draw(win, (0, 0, 0))
    win.blit(image7, (first_img_width_pos + 1, first_img_height_pos + (image_height * 2) + 40))
    image8_button.draw(win, (0, 0, 0))
    win.blit(image8, (first_img_width_pos + image_width + 20, first_img_height_pos + (image_height * 2) + 40))
    image9_button.draw(win, (0, 0, 0))
    win.blit(image9, (first_img_width_pos + image_width * 2 + 40, first_img_height_pos + (image_height * 2) + 40))
    done_button.draw(win, (0, 0, 0))
    win.blit(done_text, (first_img_width_pos + 130, first_img_height_pos + (image_height * 3) + 80))
    win.blit(category_text, (first_img_width_pos, 30))


def show_num_right(num):
    right_text = text_font.render(f'You got {num}% right', True, white)
    win.blit(right_text, (width/2 - 145, height/2 - 100))
    go_again_button.draw(win, (0, 0, 0))
    win.blit(go_again_text, (first_img_width_pos + image_width/2 + 35, height/2 - 15))
    end_button.draw(win, (0, 0, 0))
    win.blit(end_text, (first_img_width_pos + image_width * 2 + 35, height/2 - 15))


def save_image(img):
    img_id = 0
    for num in img:
        plt.imsave(f'images/image{img_id}.png', X_test[num])
        img_id += 1


def draw_window():

    if show_image:

        draw_images()

    if show_right:

        show_num_right(analysis)


text_font = pygame.font.SysFont('comicsans', 30)
instructions_font = pygame.font.SysFont('comicsans', 18)
first_img_width_pos = width/2 - button_width * 1.7
first_img_height_pos = height/2 - button_height * 2
done_button = Button((0, 0, 0), first_img_width_pos, first_img_height_pos + (image_height * 3) + 59,
                     button_width * 3 + 35, button_height)
done_text = text_font.render('Done', True, white)

main = True

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    analysis = 0.0
    image_indexes = get_image_indexes(9, X_test)
    ai_chosen_categories = ai_master_function(image_indexes)
    chosen_category = pick_category(ai_chosen_categories)
    category_name = get_category(chosen_category)
    save_image(image_indexes)

    image1 = pygame.transform.scale(pygame.image.load('images/image0.png'), (image_width, image_height))
    image1_button = Button((0, 0, 0), first_img_width_pos, first_img_height_pos, button_width, button_height)
    image2 = pygame.transform.scale(pygame.image.load('images/image1.png'), (image_width, image_height))
    image2_button = Button((0, 0, 0), first_img_width_pos + image_width + 19, first_img_height_pos, button_width,
                           button_height)
    image3 = pygame.transform.scale(pygame.image.load('images/image2.png'), (image_width, image_height))
    image3_button = Button((0, 0, 0), first_img_width_pos + image_width * 2 + 39, first_img_height_pos,
                           button_width + 2, button_height + 2)
    image4 = pygame.transform.scale(pygame.image.load('images/image3.png'), (image_width, image_height))
    image4_button = Button((0, 0, 0), first_img_width_pos, first_img_height_pos + image_height + 19, button_width,
                           button_height)
    image5 = pygame.transform.scale(pygame.image.load('images/image4.png'), (image_width, image_height))
    image5_button = Button((0, 0, 0), first_img_width_pos + image_width + 19, first_img_height_pos + image_height + 19,
                           button_width, button_height)
    image6 = pygame.transform.scale(pygame.image.load('images/image5.png'), (image_width, image_height))
    image6_button = Button((0, 0, 0), first_img_width_pos + image_width * 2 + 39,
                           first_img_height_pos + image_height + 19, button_width, button_height)
    image7 = pygame.transform.scale(pygame.image.load('images/image6.png'), (image_width, image_height))
    image7_button = Button((0, 0, 0), first_img_width_pos, first_img_height_pos + (image_height * 2) + 39, button_width,
                           button_height)
    image8 = pygame.transform.scale(pygame.image.load('images/image7.png'), (image_width, image_height))
    image8_button = Button((0, 0, 0), first_img_width_pos + image_width + 19,
                           first_img_height_pos + (image_height * 2) + 39, button_width, button_height)
    image9 = pygame.transform.scale(pygame.image.load('images/image8.png'), (image_width, image_height))
    image9_button = Button((0, 0, 0), first_img_width_pos + image_width * 2 + 39,
                           first_img_height_pos + (image_height * 2) + 39, button_width, button_height)
    go_again_button = Button((0, 0, 0), first_img_width_pos + image_width / 2 + 20, height / 2 - image_width / 2,
                             button_width, button_height)
    go_again_text = instructions_font.render('continue', True, white)
    end_button = Button((0, 0, 0), first_img_width_pos + image_width * 2, height / 2 - image_width / 2, button_width,
                        button_height)
    end_text = instructions_font.render('end', True, white)
    category_text = instructions_font.render(f'please select all images containing a(n) {category_name}', True, white)
    main = True

    while main:
        for event in pygame.event.get():
            position = pygame.mouse.get_pos()
            if event.type == pygame.QUIT:
                running = False
                main = False
                          
            if show_image:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image1_button.is_over(position) and image1_button.color == (0, 0, 0):
                        image1_button.color = (0, 255, 0)
                        user_selection.append(0)

                    elif image1_button.is_over(position) and image1_button.color == (0, 255, 0):
                        image1_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(0)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image2_button.is_over(position) and image2_button.color == (0, 0, 0):
                        image2_button.color = (0, 255, 0)
                        user_selection.append(1)

                    elif image2_button.is_over(position) and image2_button.color == (0, 255, 0):
                        image2_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(1)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image3_button.is_over(position) and image3_button.color == (0, 0, 0):
                        image3_button.color = (0, 255, 0)
                        user_selection.append(2)
                    elif image3_button.is_over(position) and image3_button.color == (0, 255, 0):
                        image3_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(2)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image4_button.is_over(position) and image4_button.color == (0, 0, 0):
                        image4_button.color = (0, 255, 0)
                        user_selection.append(3)
                    elif image4_button.is_over(position) and image4_button.color == (0, 255, 0):
                        image4_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(3)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image5_button.is_over(position) and image5_button.color == (0, 0, 0):
                        image5_button.color = (0, 255, 0)
                        user_selection.append(4)
                    elif image5_button.is_over(position) and image5_button.color == (0, 255, 0):
                        image5_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(4)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image6_button.is_over(position) and image6_button.color == (0, 0, 0):
                        image6_button.color = (0, 255, 0)
                        user_selection.append(5)
                    elif image6_button.is_over(position) and image6_button.color == (0, 255, 0):
                        image6_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(5)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image7_button.is_over(position) and image7_button.color == (0, 0, 0):
                        image7_button.color = (0, 255, 0)
                        user_selection.append(6)
                    elif image7_button.is_over(position) and image7_button.color == (0, 255, 0):
                        image7_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(6)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image8_button.is_over(position) and image8_button.color == (0, 0, 0):
                        image8_button.color = (0, 255, 0)
                        user_selection.append(7)
                    elif image8_button.is_over(position) and image8_button.color == (0, 255, 0):
                        image8_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(7)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if image9_button.is_over(position) and image9_button.color == (0, 0, 0):
                        image9_button.color = (0, 255, 0)
                        user_selection.append(8)
                    elif image9_button.is_over(position) and image9_button.color == (0, 255, 0):
                        image9_button.color = (0, 0, 0)
                        try:
                            user_selection.remove(8)
                        except ValueError:
                            print("ValueError: 2 not in list")

                if event.type == pygame.MOUSEMOTION:
                    if done_button.is_over(position):
                        done_button.color = (0, 255, 0)
                    else:
                        done_button.color = (0, 0, 0)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if done_button.is_over(position):
                        show_image = False
                        show_right = True
                        analysis = verify_choices(user_selection, chosen_category, ai_chosen_categories)

            if show_right:
                if event.type == pygame.MOUSEMOTION:
                    if go_again_button.is_over(position):
                        go_again_button.color = (0, 255, 0)
                    else:
                        go_again_button.color = (0, 0, 0)

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if go_again_button.is_over(position):
                        show_image = True
                        show_right = False
                        main = False
                        user_selection = []

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if end_button.is_over(position):
                        main = False
                        running = False

                if event.type == pygame.MOUSEMOTION:
                    if end_button.is_over(position):
                        end_button.color = (0, 255, 0)
                    else:
                        end_button.color = (0, 0, 0)
        win.fill(gray)
        draw_window()
        pygame.display.flip()
        clock.tick(60)  # limits FPS to 60

pygame.quit()
