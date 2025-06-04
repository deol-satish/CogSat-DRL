    def step(self, action):
        """
        Apply action and return (observation, reward, terminated, truncated, info)
        """

         
        terminated = False
        truncated = False

        self.currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
        self.channelFreqs = self.eng.workspace['channelFreqs']

        action = int(action)

        self.leoIndex = self.eng.workspace['leoIndex']


        if type(self.currentLEOFreqs) == type(float(0)):
            print("Action taken: ", action)
            logging.info("=== Action Taken === %s", action)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            logging.info("=== currentLEOFreqs === %s",self.currentLEOFreqs)
            self.eng.workspace['currentLEOFreqs'] = self.channelFreqs[0][action]
        else:
            currentLEOFreqs = self.eng.workspace['currentLEOFreqs']
            #Modify the values (example change)
            new_values = np.array(currentLEOFreqs).flatten().tolist()
            print("Old Frequencies: ", new_values)

            new_values[int((self.leoIndex -1))] = self.channelFreqs[0][action]

            print("New Frequencies: ", new_values)

            #Update the workspace variable
            self.eng.workspace['currentLEOFreqs'] = matlab.double(new_values)
            
            


        self.eng.eval("stepScenario", nargout=0)

        print("Step Scenario",self.eng.workspace['tIdx'])
        
        state = self.eng.workspace['snd_state']

        # Reset the observation
        next_observation = self.convert_matlab_state_to_py_state(state)
        terminated = self.eng.workspace['done']
        reward_matlab = self.eng.workspace['reward']

        reward = -32.4115512468957

        calc_reward = 0.0

        state['LEO_1']['AccessStatus']['Melbourne']
        if (state['LEO_1']['AccessStatus']['Melbourne'] and state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Melbourne']['snr'] + reward_matlab["LEO_1"]['reward']['Sydney']['snr']
            reward = reward /2
        elif (state['LEO_1']['AccessStatus']['Melbourne'] and not state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Melbourne']['snr']
        elif (not state['LEO_1']['AccessStatus']['Melbourne'] and state['LEO_1']['AccessStatus']['Sydney']):
            reward = reward_matlab["LEO_1"]['reward']['Sydney']['snr']

        # reward = math.log(reward)

        print("Reward: ", reward)
        logging.info("=== Reward === %s", reward)

        
 
        # Action: e.g., select a new frequency for one LEO
        # For this example, we let stepScenario handle the frequency selection randomly
        
        # Observation: simplified; in practice, you may compute SINR, access status, etc.
        
 
        # Reward: simplified; in practice, reward could be SINR or channel access success

        info = {"frequencies": np.array(self.currentLEOFreqs)}

        if terminated or self.eng.workspace['tIdx'] > 179:
            print("Episode terminated.")
            logging.info("=== Episode Terminated ===")
            self.eng.eval("SaveData", nargout=0)

        if self.eng.workspace['tIdx'] % 100 == 0:
            print("Saving Data every 100 steps")
            logging.info("=== Saving Data every 5 epochs ===")
            self.eng.eval("SaveData", nargout=0)
            truncated = True
 
        return next_observation, reward, terminated, truncated, info