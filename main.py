import numpy as np
import pygame
from statistics import mean
from random import randint

from player import Player
from basketball import BasketBall
from bots.simple_bot import simple_bot
from basket import Basket

from point_collider import PointCollider
from box_collider import BoxCollider

from model import Model

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
BALLS_COLLISION_SLOWDOWN = 0.8

STATS_FONT = pygame.font.Font(None, 20)
SCORE_FONT = pygame.font.Font(None, 60)
# It's better to fix the delta time for AI training for a better consistency
# And to not allow the AI to use speedups and slowdowns as a glitch strategy

bg = pygame.transform.scale_by(pygame.image.load("bg.jpg"), .4)

class Game:
    def __init__(self, fps=60):
        self.width = WIDTH
        self.height = HEIGHT

        self.analyse = False # True or False
        self.random_mode = 0 # 0 : no random, 1 : only player, 2 : all random

        # Game Constants
        self.fps = fps
        self.dt = 1 / fps
        self.width = WIDTH
        
        # Initialize game objects
        self.basket1 = Basket()
        self.basket2 = Basket()

        self.points = []
        self.points.extend(BoxCollider(0, 110, 6, 90, gap=5).generate_point_colliders())
        # First basket
        self.points += [
            PointCollider(8, 171), PointCollider(11, 171), PointCollider(13, 173),
            PointCollider(14, 176), PointCollider(14, 179), PointCollider(14, 181),
            PointCollider(16, 184), PointCollider(16, 188), PointCollider(17, 191),
            PointCollider(18, 194), PointCollider(18, 197), PointCollider(18, 201),
            PointCollider(19, 204), PointCollider(20, 209), PointCollider(20, 212),
            PointCollider(21, 216), PointCollider(21, 220), PointCollider(21, 223),
            PointCollider(21, 226), PointCollider(21, 230), PointCollider(11, 168),
            PointCollider(12, 164), PointCollider(8, 164), PointCollider(74, 172),
            PointCollider(74, 168), PointCollider(74, 164), PointCollider(77, 164),
            PointCollider(79, 164), PointCollider(79, 168), PointCollider(79, 172),
            PointCollider(76, 172), PointCollider(72, 174), PointCollider(72, 176),
            PointCollider(72, 179), PointCollider(71, 183), PointCollider(69, 186),
            PointCollider(70, 188), PointCollider(69, 191), PointCollider(68, 195),
            PointCollider(68, 199), PointCollider(67, 202), PointCollider(67, 206),
            PointCollider(66, 209), PointCollider(66, 213), PointCollider(66, 217),
            PointCollider(66, 221), PointCollider(65, 225), PointCollider(65, 228),
            PointCollider(65, 223), PointCollider(43, 210, False)
        ]
        # Second basket
        self.points.extend(BoxCollider(794, 110, 6, 90, gap=5).generate_point_colliders())
        self.points += [
            PointCollider(792, 171), PointCollider(789, 171), PointCollider(787, 173),
            PointCollider(786, 176), PointCollider(786, 179), PointCollider(786, 181),
            PointCollider(784, 184), PointCollider(784, 188), PointCollider(783, 191),
            PointCollider(782, 194), PointCollider(782, 197), PointCollider(782, 201),
            PointCollider(781, 204), PointCollider(780, 209), PointCollider(780, 212),
            PointCollider(779, 216), PointCollider(779, 220), PointCollider(779, 223),
            PointCollider(779, 226), PointCollider(779, 230), PointCollider(789, 168),
            PointCollider(788, 164), PointCollider(792, 164), PointCollider(726, 172),
            PointCollider(726, 168), PointCollider(726, 164), PointCollider(723, 164),
            PointCollider(721, 164), PointCollider(721, 168), PointCollider(721, 172),
            PointCollider(724, 172), PointCollider(728, 174), PointCollider(728, 176),
            PointCollider(728, 179), PointCollider(729, 183), PointCollider(731, 186),
            PointCollider(730, 188), PointCollider(731, 191), PointCollider(732, 195),
            PointCollider(732, 199), PointCollider(733, 202), PointCollider(733, 206),
            PointCollider(734, 209), PointCollider(734, 213), PointCollider(734, 217),
            PointCollider(734, 221), PointCollider(735, 225), PointCollider(735, 228),
            PointCollider(735, 223), PointCollider(757, 210, False)
        ]

        self.colliding = []

        self.reset()

        self.model = Model(self)
    

    def reset(self):
        if self.random_mode >= 1:
            self.player = Player(randint(100, self.width - 100), randint(50, self.height - 50), self)
            if self.random_mode == 2:
                self.basketball = BasketBall(randint(20, self.width - 80) + self.player.radius, randint(0, self.height // 3) + self.player.radius, self)
            else:
                self.basketball = BasketBall(self.width // 2, 50, self)
        else:
            self.player = Player(self.width // 2, self.height // 2, self)
            self.basketball = BasketBall(self.width // 2, 50, self)

        # Score
        self.score = 0

        self.moves_counter = 0

        self.is_gameover = False
        

    def update(self):
        # Verify if no gameover
        self.check_gameover()

        # Update position & velocities
        self.player.update()
        self.basketball.update()
        self.model.update_input()

        # Collision Handeling: Player <--> ColliderPoint
        # Find the closest collider point and compute collisions #!(NOT OPTIMISED)
        collided_points = []
        collided_points_distances = []

        for i in self.points:
            status, d = i.check_collision(self.player)
            if status:
                collided_points.append(i)
                collided_points_distances.append(d)

        if collided_points:
            closest_collided_point = collided_points[
                collided_points_distances.index(min(collided_points_distances))
            ]
            closest_collided_point.handle_collision(self.player)
            closest_collided_point.resolve_overlap(self.player)

        # Collision Handeling: BasketBall <--> ColliderPoint
        # Find the closest collider point and compute collisions #!(NOT OPTIMISED)
        collided_points = []
        collided_points_distances = []

        for i in self.points:
            status, d = i.check_collision(self.basketball)
            if status:
                collided_points.append(i)
                collided_points_distances.append(d)
            elif not i.solid:
                if i in self.colliding:
                    self.colliding.remove(i)

        if collided_points:
            closest_collided_point = collided_points[
                collided_points_distances.index(min(collided_points_distances))
            ]
            # Check if it is a solid collider
            if closest_collided_point.solid:
                closest_collided_point.handle_collision(self.basketball)
                closest_collided_point.resolve_overlap(self.basketball)
            
            # Handle collisions to score points
            elif closest_collided_point in self.colliding:
                return
            else:
                self.colliding.append(closest_collided_point)
                self.score += 1

        # Collision Handeling: Player <--> BasketBall
        if self.check_collision():
            self.handle_collison()
            self.resolve_overlap()


    def check_collision(self):
        distance = np.linalg.norm(self.player.pos - self.basketball.pos)
        if distance <= self.player.radius + self.basketball.radius:
            return True

    def resolve_overlap(self):
        # Calculate the distance vector and magnitude
        distance_vector = self.player.pos - self.basketball.pos
        distance = np.linalg.norm(distance_vector)

        # Calculate the overlap (if any)
        overlap = (self.player.radius + self.basketball.radius) - distance

        # Normalize the distance vector to get the direction of push
        direction = distance_vector / distance

        # Push both balls apart by half of the overlap
        self.player.pos += direction * (overlap / 2)
        self.basketball.pos -= direction * (overlap / 2)

    def handle_collison(self):
        # Use the Elastic Collision formula to calculate the new velocity vectors
        m1 = self.player.mass
        m2 = self.basketball.mass
        x1 = self.player.pos
        x2 = self.basketball.pos
        v1 = self.player.vel
        v2 = self.basketball.vel
        new_player_vel = (
            v1
            - (2 * m2 / (m1 + m2))
            * (v1 - v2).dot(x1 - x2)
            * (x1 - x2)
            / np.linalg.norm(x1 - x2) ** 2
        ) * BALLS_COLLISION_SLOWDOWN
        new_ball_vel = (
            v2
            - (2 * m1 / (m1 + m2))
            * (v2 - v1).dot(x2 - x1)
            * (x2 - x1)
            / np.linalg.norm(x2 - x1) ** 2
        ) * BALLS_COLLISION_SLOWDOWN
        self.player.vel = new_player_vel
        self.basketball.vel = new_ball_vel

    def play_move(self, command):
        if command != "NO JUMP":
            x, y = map(float, command.split())
            self.player.push((x, y))
        self.update()
        self.moves_counter += 1

    def fetch_data(self):
        return [
            self.player.pos[0],
            self.player.pos[1],
            self.player.vel[0],
            self.player.vel[1],
            self.basketball.pos[0],
            self.basketball.pos[1],
            self.basketball.vel[0],
            self.basketball.vel[1],
        ]

    def check_gameover(self):
        if self.basketball.pos[1] > HEIGHT + self.basketball.radius:
            self.is_gameover = True


def update_stats(real_dt, game_fps, fps_history, speed_history):
    real_fps = 1000 / real_dt
    fps_history.append(real_fps)
    speed = real_fps / game_fps
    speed_history.append(speed)

    capping_limit = 5 * real_fps
    if len(speed_history) > capping_limit:
        speed_history = speed_history[-max(int(capping_limit), 1) :]
        fps_history = fps_history[-max(int(capping_limit), 1) :]


def render_game(game):
    game_screen = pygame.Surface((WIDTH, HEIGHT))
    game_screen.blit(bg, (-WIDTH//2.4, 0))
    game.basketball.draw(game_screen)
    game.player.draw(game_screen)
    game.basket1.draw(game_screen, False)
    game.basket2.draw(game_screen, True)
    if game.analyse:
        for i in game.points:
            i.draw(game_screen)
    return game_screen


def render_ui(game, speed_history, target_game_speed, fps_history):
    ui_canvas = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

    if game.analyse:
        # Render speed and FPS in smaller font
        ui_canvas.blit(
            STATS_FONT.render(f"Speed: {round(mean(speed_history) * 100)}%", True, "White"),
            (10, 10),
        )
        ui_canvas.blit(
            STATS_FONT.render(
                f"Target Speed: {round(target_game_speed * 100)}%", True, "White"
            ),
            (10, 30),
        )
        ui_canvas.blit(
            STATS_FONT.render(f"FPS: {round(mean(fps_history))}", True, "White"),
            (10, 50),
        )
        ui_canvas.blit(
            STATS_FONT.render(f"FPS Target: {game.fps}", True, "White"),
            (10, 70),
        )
        ui_canvas.blit(
            STATS_FONT.render(f"Frame Counter: {game.moves_counter}", True, "White"),
            (10, 90),
        )

    # Render score in larger font, centered at the top of the screen
    score_text = SCORE_FONT.render(str(game.score), True, "White")
    score_rect = score_text.get_rect(
        midtop=(game.width // 2, 10)
    )  # Centered at the top
    ui_canvas.blit(score_text, score_rect)

    return ui_canvas


def play(game, window, game_speed=1):
    clock = pygame.time.Clock()
    running = True
    speed_history = []
    fps_history = []
    while running:
        # CLOCK HANDELING
        real_dt = clock.tick(game.fps * game_speed)
        update_stats(real_dt, game.fps, fps_history, speed_history)

        # EVENT HANDELING
        was_mouse_clicked = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    was_mouse_clicked = True
                else:
                    x, y = pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]
                    game.points.append(PointCollider(x, y))

        # HUMAN CONTROLS
        mouse_pos = np.array(pygame.mouse.get_pos(), dtype=float)
        if was_mouse_clicked:
            game.play_move(f"{mouse_pos[0]} {mouse_pos[1]}")
        else:
            game.play_move("NO JUMP")

        if game.is_gameover:
            game.reset()

        # RENDER
        window.blit(render_game(game), (0, 0))
        window.blit(render_ui(game, speed_history, game_speed, fps_history), (0, 0))
        pygame.display.update()


def watch_bot_play(game, ai_script, window, game_speed=1):
    clock = pygame.time.Clock()
    running = True
    speed_history = []
    fps_history = []
    while running:
        # CLOCK HANDELING
        real_dt = clock.tick(60 * game_speed)
        update_stats(real_dt, game.fps, fps_history, speed_history)

        # EVENT HANDELING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # BOT CONTROLS
        game.play_move(ai_script(game.fetch_data()))

        # RENDER
        window.blit(render_game(game), (0, 0))
        window.blit(render_ui(game, speed_history, game_speed, fps_history), (0, 0))
        pygame.display.update()


def train_model(game, window, game_speed=1, episodes=1000):
    for episode in range(episodes):
        game.reset()
        total_reward = 0
        max_steps = 500
        #print(episode)

        for step in range(max_steps):
            #print(step)
            state = game.model.update_input()
            action = game.model.model.forward(state)
            game.player.vel = action

            game.update()

            basket_pos = np.array([45, 191])
            ball_dist = np.linalg.norm(game.basketball.pos - basket_pos)
            reward = -ball_dist / 100
            total_reward += reward

            target = action + reward
            game.model.model.backward(target=target, output=action, lr=0.001)

            # RENDER (not good)
            #window.blit(render_game(game), (0, 0))
            #pygame.display.update()

        if episode % 100 == 0:
            print(f"Episode {episode}, Total Reward: {total_reward:.2f}")


    game.model.save_model("trained_model.json")


def main():
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("BasketBall AI")

    mode = "human"  # 'human' or 'bot' or 'train'
    new_game = Game()

    if mode == "human":
        # Create a new game and play it with human controls
        play(new_game, window, game_speed=1)
    elif mode == "bot":
        # Watch an AI play a new game based on ai_function
        ai_function = simple_bot
        watch_bot_play(new_game, ai_function, window, game_speed=1)
    elif mode == "train":
        # Train the model with a specified number of episodes
        train_model(new_game, window, episodes=200)


if __name__ == "__main__":
    main()