import pygame, math, random, copy, numpy

FPS = 60

screen = pygame.display.set_mode((500,500))
clock = pygame.time.Clock()

pygame.font.init()

generation = 1

class NeuralNetwork:
    def __init__(self):
        self.hidden_layer = [Neuron(6) for _ in range(8)]
        self.output = [Neuron(8) for _ in range(3)]

    def hidden_calc(self,inputs):
        a = []
        for n in self.hidden_layer:
            a.append(n.forward(inputs))
        return a
    
    def outputs(self,inputs):
        out = []
        a = self.hidden_calc(inputs)
        for n in self.output:
            out.append(n.forward(a))
        return out
    
    def mutate(self):
        for n in self.hidden_layer:
            if random.randint(1,10) == 1:
                n.mutate()
        for n in self.output:
            if random.randint(1,10) == 1:
                n.mutate()

class Neuron:
    def __init__(self,num_inputs):
        self.weights = [random.uniform(-1, 1) for _ in range(num_inputs)]
        self.bias = random.uniform(-1, 1)

    def forward(self,inputs):   
        z = 0
        for i in range(len(self.weights)):
            z += self.weights[i] * inputs[i]
        z += self.bias
        z = math.tanh(z)
        return z
    def mutate(self):
        t = random.randint(0,len(self.weights)-1) 
        self.weights[t] += random.uniform(-0.2,0.2)

        if random.randint(1,10) == 1:
            self.bias += random.uniform(-0.2,0.2)
    
class Rocket:
    def __init__(self,x,y):
        self.fitness = 0
        self.active = True

        self.x = x 
        self.y = y
        self.angle = 10
        self.turn_speed = 3
        self.width = 20
        self.height = 50

        self.fuel = 50

        self.vx = 0
        self.vy = 0

        self.gravity = 0.08
        self.thrust = 0.15

    def handle_inputs(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_d] and self.y < 400:
            self.angle += self.turn_speed

        if keys[pygame.K_SPACE] and self.fuel > 0:
            rad = math.radians(self.angle)
            
            self.vx += math.sin(rad) * self.thrust
            self.vy -= math.cos(rad) * self.thrust

            self.fuel -= 0.1

    def thrust_engine(self, power):
        if self.fuel <= 0:
            return

        rad = math.radians(self.angle)

        self.vx += math.sin(rad) * self.thrust * power
        self.vy -= math.cos(rad) * self.thrust * power

        self.fuel -= 0.1 * power

    def calculate_fitness(self,pad_x):
        pad_center =  pad_x + 50
        dx = pad_center - self.x

        angle = abs(self.get_normalised_angle())
        self.fitness = 2000

        self.fitness -= abs(dx) * 2
        self.fitness -= abs(self.vx) * 200
        self.fitness -= abs(self.vy) * 400
        self.fitness -= abs(angle) * 40

    def update(self):
        global random_x

        if not self.active:
            return

        GROUND_Y = 450 - self.height / 2
        PAD_Y = 400 - self.height / 2

        self.handle_inputs()
        self.vy += self.gravity

        self.y += self.vy
        self.x += self.vx

        if self.y >= PAD_Y and random_x < self.x < random_x + 100:
            self.y = PAD_Y

            landing_angle = self.get_normalised_angle()

            if abs(self.vy) < 1 and abs(landing_angle) <= 10:
                self.fitness += 5000
                print(f"SAFE LANDING  {self.fitness}")
            else:
                self.calculate_fitness(random_x)
                print(f"CRASH  {self.fitness}")

            self.vx = 0
            self.vy = 0

            self.active = False

        elif self.y >= GROUND_Y:
            self.calculate_fitness(random_x)
            print(f"CRASH  {self.fitness}")

            self.vx = 0
            self.vy = 0

            self.active = False

    def draw(self):
        rocket_surface = pygame.Surface((self.width, self.height),pygame.SRCALPHA)

        pygame.draw.rect(rocket_surface,(255, 0, 0),(0, 0, self.width, self.height))

        rotated = pygame.transform.rotate(rocket_surface,-self.angle)

        rect = rotated.get_rect(center=(self.x, self.y))

        screen.blit(rotated, rect)
    def get_normalised_angle(self):
        return (self.angle + 180) % 360 - 180

rocket = Rocket(240,100)

random_x = random.randint(100,400)

Font = pygame.font.SysFont(None, 40)

population_size = 50

rockets = [Rocket(240, 100) for _ in range(population_size)]

print("New rockets:", len(rockets))
print("First rocket:", rockets[0].x, rockets[0].y, rockets[0].active)

networks = [NeuralNetwork() for _ in range(population_size)]


Running = True

while Running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            Running = False

    for i in range(population_size):
        rocket = rockets[i]
        nn = networks[i]

        if not rocket.active:
            continue

        pad_center = random_x + 50
        dx = pad_center - rocket.x

        altitude = 400 - rocket.y

        inputs = [
            rocket.vx / 5,
            rocket.vy / 5,
            rocket.get_normalised_angle() / 180,
            altitude/400,
            dx / 250,
            rocket.fuel / 50
        ]

        output = nn.outputs(inputs)

        left_strength = (output[0] + 1) / 2
        right_strength = (output[1] + 1) / 2

        rocket.angle -= left_strength * rocket.turn_speed
        rocket.angle += right_strength * rocket.turn_speed

        throttle = (output[2] + 1) / 2
        rocket.thrust_engine(throttle)

        rocket.update()


    # NEW GENERATION
    if all(not rocket.active for rocket in rockets):

        best_index = max(
            range(population_size),
            key=lambda i: rockets[i].fitness
        )

        best_network = networks[best_index]

        print("Generation:", generation)
        print("Best fitness:", rockets[best_index].fitness)

        new_networks = [copy.deepcopy(best_network)]

        for _ in range(population_size - 1):
            child = copy.deepcopy(best_network)
            child.mutate()
            new_networks.append(child)

        networks = new_networks

        rockets = [
            Rocket(240, 100)
            for _ in range(population_size)
        ]

        generation += 1


    # DRAW
    screen.fill((0, 0, 0))

    pygame.draw.line(screen,(255, 255, 255),(random_x, 400),(random_x + 100, 400),2)

    pygame.draw.line(screen,(255, 255, 255),(0, 450),(500, 450),2)

    text_surface = Font.render(f"Generation: {generation}", True, (255,255,255))
    screen.blit(text_surface, (150, 462))

    for rocket in rockets:
        rocket.draw()
    pygame.display.flip()
    clock.tick(FPS)
