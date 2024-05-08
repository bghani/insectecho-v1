from config import *
from dataset import *
from inference import *
from util import *
from hear21passt.base import get_basic_model, get_model_passt

class avesecho(nn.Module):
    def __init__(self, NumClasses=585, pretrain=True, ExternalEmbeddingSize=320, hidden_layer_size=100):
        super(avesecho, self).__init__()
        self.fc1 = nn.Linear(ExternalEmbeddingSize, NumClasses)
        #self.fc2 = nn.Linear(hidden_layer_size, NumClasses)
        #self.relu = nn.ReLU()  # ReLU activation function
        #self.bn1 = nn.BatchNorm1d(NumClasses)
        

    def forward(self, x, emb):
        
        x = self.fc1(emb.squeeze(1))
       
        return x    

class AvesEcho:
    def __init__(self, model_name, slist, flist, add_filtering, mconf, maxpool, add_csv, args, outputd, avesecho_mapping):
        
        self.slist = slist
        self.flist = flist
        self.model_name = model_name
        self.add_filtering = add_filtering
        self.add_csv = add_csv
        self.mconf = mconf
        self.outputd = outputd
        self.avesecho_mapping = avesecho_mapping
        self.species_list = load_species_list(self.slist)
        self.n_classes = len(self.species_list)
        self.split_signals = split_signals
        self.maxpool = maxpool
        self.args = args

        # Load the model
        if self.model_name == 'passt':
            self.model = get_basic_model(mode = 'logits', arch="passt_s_kd_p16_128_ap486")
            self.model.net =  get_model_passt(arch="passt_s_kd_p16_128_ap486",  n_classes=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load('/home/burooj/models/avesecho_ml/best_model_passt_kld_29mar.pt', map_location=device))
        if self.model_name == 'fc':
            self.model = avesecho(NumClasses=self.n_classes)
            self.model = self.model.to(device)
            self.model.load_state_dict(torch.load('checkpoints/best_model_fc_1.pt', map_location=device))

        
    def analyze_audio(self, audio_file_path, lat=None, lon=None):
        
        # Running the model to get predictions, and then returning the results.
        
        # Starting time
        start_time = time.time()

        # Load soundfile
        sound = audio_file_path
        filtering_list = setup_filtering(lat, lon, self.add_filtering, self.flist, self.slist)

        if not os.path.exists(self.outputd):
            os.makedirs(self.outputd)

        # Split signal into 3s chunks
        self.split_signals(sound, self.outputd, signal_length=3, n_processes=10)


        # Extract the filename from the path
        filename = sound.split('/')[-1]  # This splits the string by '/' and gets the last element
        

        #Load a list of files for in a dir
        inference_dir = self.outputd
        inference_data = [os.path.join(inference_dir, f) for f in sorted(os.listdir(inference_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

        #Inference
        inference_set = InferenceDataset(inference_data, self.n_classes, model=self.model_name)
        params_inf = {'batch_size': 64, 'shuffle': False, 'num_workers': 5}
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        # Maps species common names to scientific names and also across XC and eBird standards and codes
        df = pd.read_csv(self.avesecho_mapping, header=None, names=['ScientificName', 'CommonName'])


        output = {
        "generated_by": {
            "datetime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tag": "avesecho-v1",
            "version": "1fd68f8c8cb93ec4e45049fcf9a056628e9599aa815790a2a7b568aa"
        },
        "media": [],
        "region_groups": [],
        "predictions": []
        }

        if self.maxpool:
            predictions, scores, files = inference_maxpool(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
            output = create_json_maxpool(output, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf, len(inference_data))
        else:
            predictions, scores, files = inference(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
            output = create_json(output, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf)
        

        # Determine the output file name based on filtering
        json_name = f'outputs/analysis-results.json'
    
        # Write the output dictionary to a JSON file
        with open(json_name, 'w') as json_file:
            json.dump(output, json_file, indent=4)
            
        #Compute the elapsed time in seconds
        elapsed_time = time.time() - start_time
        
        # Print the result
        print(f"It took {elapsed_time:.2f}s to analyse {filename}.")

        # Empty temporary audio chunks directory
        shutil.rmtree(self.outputd)

    def analyze_batch(self, audio_dir_path, lat=None, lon=None):
        
        # Running the model to get predictions, and then returning the results.

      
        filtering_list = setup_filtering(lat, lon, self.add_filtering, self.flist, self.slist)


        output = {
        "generated_by": {
            "datetime": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tag": "avesecho-v1",
            "version": "1fd68f8c8cb93ec4e45049fcf9a056628e9599aa815790a2a7b568aa"
        },
        "media": [],
        "region_groups": [],
        "predictions": []
        }
        
        for audio_file in os.listdir(audio_dir_path):

            if not (audio_file.endswith('.wav') or audio_file.endswith('.mp3') or audio_file.endswith('.WAV') or audio_file.endswith('.ogg') or audio_file.endswith('.flac') ):
                continue  # Skip the rest of the loop for this iteration

              # Load soundfile
            sound = os.path.join(audio_dir_path, audio_file)

            if not os.path.exists(self.outputd):
                os.makedirs(self.outputd)

            # Split signal into 3s chunks
            self.split_signals(sound, self.outputd, signal_length=3, n_processes=10)


            # Extract the filename from the path
            filename = sound.split('/')[-1]  # This splits the string by '/' and gets the last element
            

            #Load a list of files for in a dir
            inference_dir = self.outputd
            inference_data = [os.path.join(inference_dir, f) for f in sorted(os.listdir(inference_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

            #Inference
            inference_set = InferenceDataset(inference_data, self.n_classes, model=self.model_name)
            params_inf = {'batch_size': 64, 'shuffle': False, 'num_workers': 5}
            inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

            # Maps species common names to scientific names and also across XC and eBird standards and codes
            df = pd.read_csv(self.avesecho_mapping, header=None, names=['ScientificName', 'CommonName'])


            if self.maxpool:
                predictions, scores, files = inference_maxpool(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
                output = create_json_maxpool(output, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf, len(inference_data))
            else:
                predictions, scores, files = inference(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
                output = create_json(output, predictions, scores, files, self.args, df, self.add_csv, filename, self.mconf)

                
            # Empty temporary audio chunks directory
            shutil.rmtree(self.outputd)
        
        # Determine the output file name based on filtering
        json_name = f'outputs/analysis-results.json'
    
        # Write the output dictionary to a JSON file
        with open(json_name, 'w') as json_file:
            json.dump(output, json_file, indent=4)
        
        
    def analyze_warblr_audio(self, audio_file_path, lat=None, lon=None):
        
        # Running the model to get predictions, and then returning the results.
        
        # Starting time
        start_time = time.time()

        # Load soundfile
        sound = audio_file_path
        filtering_list = setup_filtering(lat, lon, self.add_filtering, self.flist, self.slist)

        if not os.path.exists(self.outputd):
            os.makedirs(self.outputd)

        # Split signal into 3s chunks
        self.split_signals(sound, self.outputd, signal_length=3, n_processes=10)


        # Extract the filename from the path
        filename = sound.split('/')[-1]  # This splits the string by '/' and gets the last element
        

        #Load a list of files for in a dir
        inference_dir = self.outputd
        inference_data = [os.path.join(inference_dir, f) for f in sorted(os.listdir(inference_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))]

        #Inference
        inference_set = InferenceDataset(inference_data, self.n_classes, model=self.model_name)
        params_inf = {'batch_size': 64, 'shuffle': False, 'num_workers': 5}
        inference_generator = torch.utils.data.DataLoader(inference_set, **params_inf)

        # Maps species common names to scientific names and also across XC and eBird standards and codes
        df = pd.read_csv(self.avesecho_mapping, header=None, names=['ScientificName', 'CommonName'])

        # Run the inference
        predictions, scores, files = inference_warbler(self.model, inference_generator, device, self.species_list, filtering_list, self.mconf)
            
        #Compute the elapsed time in seconds
        elapsed_time = time.time() - start_time
        
        # Print the result
        print(f"It took {elapsed_time:.2f}s to analyse {filename}.")

        # Empty temporary audio chunks directory
        shutil.rmtree(self.outputd)
        
        return predictions, scores
