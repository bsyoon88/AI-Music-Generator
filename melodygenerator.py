import json
import tensorflow as tf
import tensorflow.keras as keras
from preprocess import SEQUENCE_LENGTH,MAPPING_PATH
import numpy as np
import music21 as m21

class MelodyGenerator:
  
    def __init__(self,model_path="model.h5"):
        
        self.model_path=model_path
        self.model=keras.models.load_model(model_path)
        
        with open(MAPPING_PATH,'r') as fp:
            
            
            self._mapping = json.load(fp)
            
            
        self._start_symbol = ["/"]*SEQUENCE_LENGTH
        
        
        
    def generator_melody(self , seed, num_steps, max_sequence_length,temperature):
        #Seed example :- "64","_","67","_"
        #num_steps :- number of steps in time series representation
        #max_sequence_length :- how many steps needed in a seed 
        
        #create seed with start symbol
        
        seed = seed.split()
        melody = seed
        seed = self._start_symbol + seed
        
        #map seed to int
        seed = [self._mapping[symbol] for symbol in seed]
        
        
        for _ in range (num_steps):
            
            #limit the seed to max_sequence_length
            seed = seed[-max_sequence_length:]
            
            #one-hot encode the seed 
            onehot_seed = keras.utils.to_categorical(seed , num_classes = len(self._mapping))
            #(1,max_squence_length, num of symbols in vocabulary)
            
            onehot_seed = onehot_seed[np.newaxis , ...]

            #make a prediction
             
            probabilities = self.model.predict(onehot_seed)[0]
             # [0.1, 0.2, 0.1, 0.6] -> 1
             
            output_int = self._sample_with_temperature(probabilities,temperature)
             
             #update seed
            seed.append(output_int)
             
             
             #map int to our encoding
            output_symbol = [k for k, v  in self._mapping.items() if v== output_int][0]
            
            # check whether we're at end of the melody
            
            if output_symbol == "/":
                break
            #update the melody
            melody.append(output_symbol)
            
        return melody




    def _sample_with_temperature(self,probabilities,temperature):
        #temperature -> infinity 
        #temperature -> 0 
        #temperatute -> 1
        
        prediction = np.log(probabilities)/temperature
        probabilities = np.exp(prediction)/np.sum(np.exp(prediction))

        choices = range(len(probabilities))
        index = np.random.choice(choices,p=probabilities)        
        
        return index
    
    
    
    def save_melody(self , melody, step_duration=0.25 , format="midi", file_name="mel.midi"):
        
        # create a music21 stream
        stream = m21.stream.Stream()
        
        #parse all the symbol in the melody and create note/rest objects
        #60 _ _ _ r _ 62 _
        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):

            # handle case in which we have a note/rest
            if symbol != "_" or i + 1 == len(melody):

                # ensure we're dealing with note/rest beyond the first one
                if start_symbol is not None:

                    quarter_length_duration = step_duration * step_counter # 0.25 * 4 = 1

                    # handle rest
                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    # handle note
                    else:
                        m21_event = m21.note.Note(int(start_symbol), quarterLength=quarter_length_duration)

                    stream.append(m21_event)

                    # reset the step counter
                    step_counter = 1

                start_symbol = symbol

            # handle case in which we have a prolongation sign "_"
            else:
                step_counter += 1

        # write the m21 stream to a midi file
        stream.write(format, file_name)

        
        
        
        
        
    
if __name__ == "__main__" :
  
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
    
    mg=MelodyGenerator() 
    seed ="55 _ _ _ 60 _ _ _ 60 _ _ _ 62 _"
    seed2 = "55 _ 55 _ 60 _ 64 _ 67 _"
    melody = mg.generator_melody(seed, 500, SEQUENCE_LENGTH, 0.3)
    print (melody)
    mg.save_melody(melody,file_name="newsong.midi")
    