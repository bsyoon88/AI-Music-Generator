import os
import music21 as m21
import json

KERN_DATASET_PATH="deutschl/test"
ACCEPTABLE_DURATION=[
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2,
    3,
    4
]

SAVE_DIR='dataset'
SINGLE_FILE_DATASET="file_dataset"
MAPPING_PATH="mapping.json"
SEQUENCE_LENGTH=64



def load_songs_in_kern(dataset_path):

    songs=[]
    #go through all the files in dataset and load them with music21
    for path,subdirs,files in os.walk(dataset_path):
        for file in files :
            if file[-3:]=='krn':
           
                song=m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs             



def has_acceptable_duration(song,acceptable_duration):
    for note in song.flat.notesAndRests:
        if note.duration.quarterLength not in acceptable_duration:
            return False
    return True



def transpose(song):
    #get key from score/song
    parts =song.getElementsByClass(m21.stream.Part)
    measure_part0=parts[0].getElementsByClass(m21.stream.Measure)
    key = measure_part0[0][4]

    #estimate key using music21
    if not isinstance(key,m21.key.Key):
        key = song.analyze("key")


    #get/calculate interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode =='major':
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch('C'))
    elif key.mode=='minor':
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch('A'))

    #transpose song by calulated interval
    transposed_song=song.transpose(interval)


    return transposed_song





def encode_song(song,time_step=0.25):
    # p=60,d=1.0 ->[60,"_","_","_"]

    encoded_song=[]

    for event in song.flat.notesAndRests:
        
        #handling notes
        if isinstance(event,m21.note.Note):
            symbol=event.pitch.midi #60
        #handling rest 
        elif isinstance(event,m21.note.Rest):
            symbol="r"

        #convert the note/rest into time series notation
        steps= int(event.duration.quarterLength/time_step)   
        for step in range(steps):
            if step==0:
                encoded_song.append(symbol)
            else:
                encoded_song.append("_")

    #cast encoded song to str
    encoded_song=' '.join(map(str,encoded_song))

    return encoded_song



def load(file_path):
    with open(file_path,'r') as fp:
        song=fp.read()
    return song









def preprocess(dataset_path):   
    pass


    #load the folk songs 
    print('Loading songs....')
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")


    for i,song in enumerate(songs):

        #filter out songs that have non-acceptable duration
        if not has_acceptable_duration(song,ACCEPTABLE_DURATION):
            continue

        #transposme songs to Caj/Amin
        song= transpose(song)

        #encode songs with music time series representaion
        encoded_song=encode_song(song)

        #save song to text file
        save_path=os.path.join(SAVE_DIR, str(i))
        with open(save_path,'w') as fp:
            fp.write(encoded_song)






def create_single_file_dataset(dataset_path, file_dataset_path,sequence_length):

    new_song_delimiter='/ ' *sequence_length
    songs=''
    #load encoded songs and add delimiters
    for path, _,files in os.walk(dataset_path):
        for file in files:
            file_path=os.path.join(path,file)
            song=load(file_path)
            songs=songs + song + " " +new_song_delimiter

    songs=songs[:-1]

    #save string that contain all dataset
    with open(file_dataset_path,'w') as fp:
        fp.write(songs)

    return songs 




def create_mapping(songs,mapping_path):
    mappings={}
    #identify the vocabulary
    songs=songs.split()
    vocabulary=list(set(songs))

    #create mapping
    for i,symbol in enumerate(vocabulary):
        mappings[symbol]= i
    
    #save vocabulaty in JSON file
    with open(mapping_path,'w') as fp:
        json.dump(mappings,fp,indent=4)




def main():
    preprocess(KERN_DATASET_PATH)
    songs= create_single_file_dataset(SAVE_DIR,SINGLE_FILE_DATASET,SEQUENCE_LENGTH)
    create_mapping(songs,MAPPING_PATH)


   
if __name__=="__main__":
    main()