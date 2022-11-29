from importlist import *

def normalize(X):

    X_norm = X / np.linalg.norm(X,axis=1,ord=2,keepdims= True)

    return X_norm

DataList = ["data/winequality-red.csv", 
            "../large_dataset/ground_truth.csv"]

class csv2arrays:

    def redWine(data2train):
        data = pd.read_csv(DataList[data2train])

        # replace quality data
        data['quality'].replace(to_replace={3:0,4:0,5:0,6:0,7:1,8:1}, inplace = True)

        X = data[['fixed acidity',
                'volatile acidity',
                'citric acid',
                'residual sugar',
                'chlorides',
                'free sulfur dioxide',
                'total sulfur dioxide',
                'density',
                'pH',
                'sulphates',
                'alcohol']]

        Y = data['quality']

        X = (X.values)
        Y = (Y.values)

        return X,Y
    
    def tomAndJerry(data2train):
        data = pd.read_csv(DataList[data2train])

        data_imgSetLoc = "../large_dataset/tom_and_jerry/tom_and_jerry/combined/"

        # Check data
        #print(data.head())

        # Information about the data columns
        #print(data.info())


        dataset_dict = {}

        # (1,5478) -> (imageArray, number of examples)
        #X = data[['filename']].T
        # (2,5478) 
        Y = data[['tom','jerry']]
        Y = Y.values
        #print("from mini batching")
        print(Y.shape)


        # file_list = os.listdir(data_imgSetLoc)

        # file_list.sort(key= lambda i: int(i.lstrip('frame').rstrip('.jpg')))


        # index = 0

        # for file in file_list:
        #     comp_path = Image.open(data_imgSetLoc + file)
        #     comp_path = comp_path.resize((256,144))#((854,480))

            

        #     toAdd = asarray(comp_path)

        #     # to visualize
        #     # plt.imshow(toAdd, interpolation='nearest')
        #     # plt.show()

        #     # from (1, -1) to (-1, 1)
        #     toAdd = np.reshape(toAdd, (-1,1))

        #     print("image dim: ", toAdd.shape)

        #     if index < 1:
        #         index += 1
        #         X = copy.deepcopy(toAdd)

        #     else:
        #         X = np.append(X, toAdd, axis=1)
              
        #     print(file)
        #     print("new X shape: ", X.shape)

        # prepare dataset file
        # h5f = h5py.File('TomAndJerryDataset.h5','w')
        # h5f.create_dataset('dataset_1', data = X)
        # h5f.close()

        #load dataset from a file
        h5f_load = h5py.File('../large_dataset/TomAndJerryDataset.h5','r')
        X = h5f_load['dataset_1'][:]
        h5f_load.close()

        return X,Y


DataList_funcs = [
                    csv2arrays.redWine, 
                    csv2arrays.tomAndJerry
                 ]


def initialize_dataset(data2train, run_normalize = False, test_size = 0.2):

    X, Y = DataList_funcs[data2train](data2train)

    if run_normalize:
        X = normalize(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=test_size)

    X_train = X_train.T
    Y_train = (np.reshape(Y_train,(Y_train.shape[0],-1))).T
    X_test = X_test.T
    Y_test  = (np.reshape(Y_test,(-1,1))).T


    Dataset = {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
    }

    return Dataset