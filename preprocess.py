import os
import music21 as m21

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

    print(key)
    #get/calculate interval for transposition. E.g., Bmaj -> Cmaj
    if key.mode =='major':
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch('C'))
    elif key.mode=='minor':
        interval = m21.interval.Interval(key.tonic,m21.pitch.Pitch('A'))

    #transpose song by calulated interval
    transposed_song=song.transpose(interval)


    return transposed_song





def preprocess(dataset_path):   
    pass


    #load the folk songs 
    print('Loading songs....')
    songs=load_songs_in_kern(dataset_path)
    print(f"Loaded {len(songs)} songs.")


    for song in songs:

        #filter out songs that have non-acceptable duration
        if not has_acceptable_duration(song,ACCEPTABLE_DURATION):
            continue

         #transpose songs to Cmaj/Amin
        song= transpose(song)



    #encode songs with music time series representaion

    #save song to text file


if __name__=="__main__":
    songs=load_songs_in_kern(KERN_DATASET_PATH)
    print(f"Loaded {len(songs)} songs")
    print(songs)
    song=songs[0]

    print(f"Has acceptable duration?{has_acceptable_duration(song,ACCEPTABLE_DURATION)}")
    transposed_song=transpose(song)

    
    transposed_song.show()
    