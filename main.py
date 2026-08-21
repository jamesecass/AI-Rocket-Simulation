import pygame, math, random, copy, numpy

FPS = 60

screen = pygame.display.set_mode((500,500),pygame.NOFRAME)
clock = pygame.time.Clock()

pygame.font.init()

generation = 1

class NeuralNetwork:
    def __init__(self,input_size):
        self.weights1 = numpy.random.uniform(-1,1,(input_size,8)) # Initialising Hidden layers weights
        self.bias1 = numpy.random.uniform(-1,1,(1,8)) # Hiddens Bias

        self.weights2 = numpy.random.uniform(-1, 1, (8, 2)) # Output Layers weights
        self.bias2 = numpy.random.uniform(-1, 1, (1, 2)) # Outputs Bias

    def forward_prop(self,inputs):
        self.z1 = numpy.dot(inputs,self.weights1) + self.bias1 # Matrix Multiplication for Hidden layer
        self.a1 = numpy.tanh(self.z1)

        self.a2 = numpy.dot(self.a1,self.weights2) + self.bias2 # Matrix Multiplication for output layer
        self.a2 = numpy.tanh(self.a2) # Activation Function
        return self.a2
         
    def mutate(self,Mutation_scale):
        mask = numpy.random.uniform(0,1,(self.weights1.shape))
        mutation = numpy.random.uniform(-Mutation_scale,Mutation_scale,(self.weights1.shape))
        self.weights1 = numpy.where(mask < 0.05,mutation + self.weights1,self.weights1) # Mutation for hidden Weights

        mask = numpy.random.uniform(0,1,(self.bias1.shape))
        mutation = numpy.random.uniform(-Mutation_scale,Mutation_scale,(self.bias1.shape))
        self.bias1 = numpy.where(mask < 0.05,mutation + self.bias1,self.bias1) # Mutation for hidden Bias

        mask = numpy.random.uniform(0,1,(self.weights2.shape))
        mutation = numpy.random.uniform(-Mutation_scale,Mutation_scale,(self.weights2.shape))
        self.weights2 = numpy.where(mask < 0.05,mutation + self.weights2,self.weights2) #Mutation for Output Weights
        
        mask = numpy.random.uniform(0,1,(self.bias2.shape))
        mutation = numpy.random.uniform(-Mutation_scale,Mutation_scale,(self.bias2.shape))
        self.bias2 = numpy.where(mask < 0.05,mutation + self.bias2,self.bias2) #Mutation for Output Bias
    
class Rocket:
    def __init__(self,x,y):
        self.fitness = 0
        self.active = True

        self.x = x 
        self.y = y
        self.angle = 0
        self.turn_speed = 6
        self.width = 20
        self.height = 50

        self.fuel = 50

        self.vx = 0
        self.vy = 0

        self.gravity = 0.08
        self.thrust = 0.15

    def thrust_engine(self, power):
        if self.fuel <= 0:
            return
        # Function for Neural Nets thrust output
        rad = math.radians(self.angle)

        self.vx += math.sin(rad) * self.thrust * power
        self.vy -= math.cos(rad) * self.thrust * power

        self.fuel -= 0.1 * power

    def calculate_fitness(self,pad_x):
        pad_center =  pad_x + 50
        dx = pad_center - self.x
        angle = abs(self.get_normalised_angle())

        self.fitness = 5000

        proximity = max(0, 1 - abs(dx) / 200)
        self.fitness -= abs(self.vx) * (200 + proximity * 800)

        self.fitness -= abs(self.vy) * 300
        self.fitness -= angle * 30
        self.fitness -= abs(dx) * 10

    def update(self,random_x):
        if not self.active:
            return

        GROUND_Y = 450 - self.height / 2
        PAD_Y = 400 - self.height / 2

        self.vy += self.gravity

        self.y += self.vy
        self.x += self.vx

        if self.y >= PAD_Y and random_x < self.x < random_x + 100:
            self.y = PAD_Y

            landing_angle = self.get_normalised_angle()

            if abs(self.vy) < 1 and abs(landing_angle) <= 20:
                pad_center = random_x + 50
                landing_precision = abs(pad_center - self.x) 

                landing_angle = abs(self.get_normalised_angle())
                landing_speed = abs(self.vy)

                landing_bonus = 5000

                landing_bonus -= landing_precision * 40   
                landing_bonus -= landing_angle * 100      
                landing_bonus -= landing_speed * 1000      

                landing_bonus += self.fuel * 20     

                self.fitness += max(landing_bonus, 1000)
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

def get_rocket_fitness(index):
    return rockets[index].fitness

rocket = Rocket(240,100)

random_x = random.randint(100,400)

Font = pygame.font.SysFont(None, 40)

population_size = 50

rockets = [Rocket(240, 100) for _ in range(population_size)]

print("New rockets:", len(rockets))
print("First rocket:", rockets[0].x, rockets[0].y, rockets[0].active)

networks = [NeuralNetwork(6) for _ in range(population_size)]

Mutation_scale = 0.5
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

        inputs = numpy.array([rocket.vx / 5,rocket.vy / 5,rocket.get_normalised_angle() / 180,altitude/400,dx / 250,rocket.fuel / 50])

        output = nn.forward_prop(inputs)
        output = output.tolist()

        turn = output[0][0]

        rocket.angle -= turn * rocket.turn_speed

        throttle = (output[0][1] + 1) / 2
        rocket.thrust_engine(throttle)

        rocket.update(random_x)


    # NEW GENERATION
    if all(not rocket.active for rocket in rockets):
        sorted_indices = sorted(range(population_size), key=get_rocket_fitness, reverse=True)

        # Top 5 Neural Nets
        top_5_indices = sorted_indices[:5]

        # Print Highest score
        best_index = top_5_indices[0]
        print("Generation:", generation)
        print("Best fitness:", rockets[best_index].fitness)

        new_networks = []

        for i in top_5_indices:
            new_networks.append(networks[i])

        for _ in range(population_size-5):
            parent_index = random.choice(top_5_indices)
            child = copy.deepcopy(networks[parent_index])
            child.mutate(Mutation_scale)
            new_networks.append(child)

        networks = new_networks

        rockets = [Rocket(240, 100)for _ in range(population_size)]
        random_x = random.randint(100, 400)

        generation += 1
        Mutation_scale = max(Mutation_scale - 0.01,0.02)
        print(f"{Mutation_scale:2f}")


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
    
