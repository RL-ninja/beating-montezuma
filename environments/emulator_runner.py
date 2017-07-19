from multiprocessing import Process


class EmulatorRunner(Process):

    def __init__(self, worker_id, emulators, variables, queue, barrier):
        super(EmulatorRunner, self).__init__()
        self.worker_id = worker_id
        self.emulators = emulators
        self.variables = variables
        self.queue = queue
        self.barrier = barrier

    def run(self):
        super(EmulatorRunner, self).run()
        self._run()

    def _run(self):
        count = 0
        while True:
        
            instruction = self.queue.get()
            if instruction is None:
                break
            
            shared_states = self.variables[0]
            shared_rewards = self.variables[1]
            shared_episode_over = self.variables[2]
            shared_actions = self.variables[3]
            
            for i, (emulator, action) in enumerate(zip(self.emulators, shared_actions)):            
                emulator = self.emulators[i]
                new_s, reward, episode_over = emulator.next(action)
                if episode_over:
                    shared_states[i] = emulator.get_initial_state()
                else:
                    shared_states[i] = new_s            
                shared_rewards[i] = reward
                shared_episode_over[i] = episode_over
            
            count += 1
            
            # barrier is a queue shared by all workers
            # when a worker is done executing actions for envs it manages
            # it puts True to barrier which later should be 
            self.barrier.put(True)



