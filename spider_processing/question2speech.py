from boto3 import Session
from botocore.exceptions import BotoCoreError, ClientError
from contextlib import closing
import os
import sys
from tqdm import tqdm

def speech_generation(in_txt_path, out_dir_path):
    os.makedirs(out_dir_path, exist_ok=True)

    # Create a client using the credentials and region defined in the [adminuser]
    # section of the AWS credentials file (~/.aws/credentials).
    session = Session(profile_name="adminuser")
    polly = session.client("polly")

    with open(in_txt_path, 'r') as f:
        questions = f.read().strip().split('\n')

    for i, question in tqdm(enumerate(questions), total=len(questions)):
        try:
            # Request speech synthesis
            response = polly.synthesize_speech(Text=question, OutputFormat="mp3", VoiceId="Joanna")
        except (BotoCoreError, ClientError) as error:
            # The service returned an error, exit gracefully
            print(error)
            sys.exit(-1)

        # Access the audio stream from the response
        if "AudioStream" in response:
            # Note: Closing the stream is important because the service throttles on the
            # number of parallel connections. Here we are using contextlib.closing to
            # ensure the close method of the stream object will be called automatically
            # at the end of the with statement's scope.
            with closing(response["AudioStream"]) as stream:
                output = os.path.join(out_dir_path, "{}.mp3".format(i))

                try:
                    # Open a file for writing the output as a binary stream
                    with open(output, "wb") as file:
                        file.write(stream.read())
                except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    sys.exit(-1)

        else:
            # The response didn't contain audio data, exit gracefully
            print("Could not stream audio")
            sys.exit(-1)

if __name__ == '__main__':
    spider_path = '/Users/mac/Desktop/syt/Deep-Learning/Dataset/spider'
    train_txt_path = os.path.join(spider_path, 'my', 'train', 'question.txt')
    train_speech_dir = os.path.join(spider_path, 'my', 'train', 'speech_audios')
    dev_txt_path = os.path.join(spider_path, 'my', 'dev', 'question.txt')
    dev_speech_dir = os.path.join(spider_path, 'my', 'dev', 'speech_audios')

    speech_generation(train_txt_path, train_speech_dir)




