import numpy as np
import copy 
from sklearn import preprocessing
import pandas as pd
import random
import scipy 


#TODO compare dice and own algorithm by metrics. Update: own algorithm perfoms better with own metrics. this was expected. need to recreate dice metrics
#TODO private data interface
#TODO add option for inserting manual entry
#TODO perform grid or random search for hyperparameters like pop_size or elite_count.... 
class CFGenerator:
    def __init__(self,cf_instance,desired_range,cf_data,outcome_func,elite_count=50,pop_size=300,
                max_iterations = 500, threshold = 0.05,raw_to_output=True,features_ignored=None,significant_places=1):
        self.outcome_func = outcome_func
        #must be array
        self.desired_range =  desired_range
        if isinstance(cf_instance,int):
            self.original = cf_data[cf_instance]
        else:
            self.original = cf_data.to_raw_format(cf_instance)
            pass
        self.data = cf_data
        
        #number of selected top candidates in each run
        self.elite_count= elite_count
        #count of copied original instances
        self.pop_size = pop_size
        #maximum iterations of algorithm before stopping
        self.max_iterations = max_iterations
        #max distance between two generation fitnesses before starting countdown to stop
        self.threshold = threshold
        self.feature_length = len(cf_data.data.T)
        
        #wether to pass the data in original format or internal data format
        self.raw_to_output = raw_to_output
        
        #number of significant places of floats. prevents candidates that only have epsilon distance
        if isinstance(significant_places,int):
            self.significant_places = significant_places
        else:
            raise SystemError("Not implemented")
        
        self.features_to_vary = np.array(range(self.feature_length-1))
        
        #which features to ignore while running alogrithm
        #IDEA multiple runs in which some features are ignored (different in each run) for truly diverse CFs
        if features_ignored != None:
            for i in features_ignored:
                x = self.data.column_names.index(i)
                self.features_to_vary = np.delete(self.features_to_vary,x)
        np.seterr(divide='ignore')
    
    #main loop
    def generate_counterfactuals(self,cf_count):
        #useful for debugging. makes graph after run possible
        self.plot_y = []
        self.plot_x = []
        self.i=0
        
        #setting random seed makes algorithm truly stable 
        #np.random.seed(2)
        #random.seed(2)
        
        population = np.full((self.pop_size,len(self.original)),self.original)
        population = population.astype(float)
        self.mutate_and_mate(population,0,0)
        
        iterations = 0
        previous_best_loss = -np.inf
        current_best_loss = np.inf
        stop_cnt = 0
        num_cfs = 0
        while iterations<self.max_iterations:
            #if distance of losses in each generation is <threshold start counting countdown. if distance stays smaller for 10 iterations stop algorithm 
            if abs(previous_best_loss - current_best_loss) <= self.threshold and num_cfs >= cf_count:
                stop_cnt+=1
            else:
                stop_cnt=0
            if stop_cnt >=10:
                break;
            
            previous_best_loss = current_best_loss
            self.mutate_and_mate(population,cf_count,0.6)
            old_population = np.unique(population,axis=0)
            fitness = self.calculate_loss(old_population)
            num_cfs = np.sum(np.greater_equal(50,fitness))
            indices = np.argsort(fitness)
            old_population = old_population[indices]
            fitness = fitness[indices]
            current_best_loss = np.sum(fitness[0:cf_count])
                        
            self.plot_y.append(current_best_loss)
            self.plot_x.append(self.i)
            self.i+=1

            elite_index=0
            temp_elite_count = self.elite_count if len(old_population)>self.elite_count else len(old_population)
            #distribute best candidates among population
            for i in range(0, self.pop_size):
                population[i] = np.copy(old_population[elite_index])
                elite_index += 1
                if elite_index== temp_elite_count:
                    elite_index=0
            iterations+=1
        counterfactuals = old_population[0:cf_count]
        self.raw_cf = population[0:cf_count]
        self.raw_pop = population
        return self.decode(counterfactuals)
    
    #either mutate (change one feature) or mate (exchange one feature with another point)
    def mutate_and_mate(self, datapoints,cf_count,mate_prob):
        change_index = np.random.choice(self.features_to_vary,size=len(datapoints))
        for i in range(cf_count,len(change_index)):
            prob = random.random()
            if prob > mate_prob:
                    upper = self.data.range[change_index[i]][1]
                    lower = self.data.range[change_index[i]][0]
                    if change_index[i] in self.data.cat_indices or change_index[i] in self.data.int_indices:
                        datapoints[i][change_index[i]] = np.random.randint(lower,upper+1)
                    else:
                        datapoints[i][change_index[i]] = np.round(np.random.uniform(lower,upper),self.significant_places)
            else:
                parent2 = np.random.randint(len(datapoints))
                datapoints[i][change_index[i]] = datapoints[parent2][change_index[i]] 
        return datapoints
    
    
    def calculate_loss(self,datapoints):
        datapoints = np.copy(datapoints)
        
        l1_weight = 1
        l2_weight = 1
        diversity_weight = 0
        
        #when calculating loss categorical data is transformed to 1 (different from original) and 0
        #this is a factor which can be applied which rewards or punishes changing categoricals.
        #TODO add the option to transform categorical column to int column to allow different level of difficulty 
        #i.e. from bachelor to master is easier than from school to bachelor
        cat_fac = 0.4
        
        #wether 
        loss = self.y_loss(datapoints)
        loss += diversity_weight*self.diversity_loss(datapoints)
        
        #print(datapoints)
        #set categorical variables to either cat_fac (when different) or 0 for correct calculation of norm
        datapoints= datapoints.transpose()
        i = (np.not_equal(self.original[self.data.cat_indices],datapoints[self.data.cat_indices].transpose()).astype(float)*cat_fac).transpose()
        datapoints[self.data.cat_indices]=i
        datapoints=datapoints.transpose()
        #end
        
        loss += l1_weight*self.distance_loss(datapoints,self.data.feature_weights,1)
        loss += l2_weight*self.distance_loss(datapoints,self.data.no_weight,2)
        return loss
    
    #wether datapoint is different or not from original
    def y_loss(self,datapoints):
        if self.raw_to_output:
            x = self.outcome_func(datapoints)
        else:
            data_temp = self.data.backtransform_point(datapoints)
            x = self.outcome_func(data_temp)
        yloss = 1/(np.logical_and(x >= self.desired_range[0],x <= self.desired_range[1])).astype(float)-1
        yloss[yloss==np.inf]=100
        return yloss
    
    #how many features were changed
    def diversity_loss(self,datapoints,inverse=False):
        div_loss = None
        if inverse:
            div_loss = np.sum(np.equal(self.original,datapoints),axis=1)
        else:
            div_loss = np.sum(np.not_equal(self.original,datapoints),axis=1)
        return div_loss/len(self.original)
    
    #normalized l^x norm between datapoints and original
    def distance_loss(self,datapoints,weights,order):

        x1hat = (datapoints-self.data.range.T[0])/self.data.span
        x2hat = (self.original-self.data.range.T[0])/self.data.span
        x2hat[self.data.cat_indices]=0

        dist = np.linalg.norm(weights*(x1hat-x2hat),order,axis=1)
        #TODO there is something strange going on here
        #more examples of CF datasets are needed to find out why this causes failure sometimes
        dist /= np.sum(weights)
        return dist
    
    #convert table in internal data format to pandas dataframe with "-" where original and datapoint are same
    def decode(self,cfs):    
        cfs = np.concatenate((np.array([self.original]),cfs),axis=0)
        if self.raw_to_output:
            outcome = self.outcome_func(cfs)
        else:
            data_temp = self.data.backtransform_point(cfs)
            outcome = self.outcome_func(data_temp)
        eq = []
        for i in cfs:
            eq.append(np.equal(self.original,i))
        counterfactuals = self.data.backtransform_point(cfs)
        for i in range(len(cfs[0])):
            for j in range(1,len(cfs)):
                if eq[j][i] == True:
                    counterfactuals.iloc[j,i]="-"
        counterfactuals[self.data.outcome] = outcome
        return counterfactuals
    

#holds metadata about data like ranges, MAD, which columns are int etc...
class CFData:
    def __init__(self,data,outcome):
        #prevent changes in original dataset
        data = data.copy(deep=True)
        
        self.outcome = outcome
        self.column_names = list(data.columns)
        
        #which columns are categorical data
        self.categorical_columns = data.select_dtypes(exclude=[np.number])
        self.cat_indices = [data.columns.get_loc(i) for i in self.categorical_columns]
        
        #encode categorical data
        self.categorical_encoders = []
        for i in self.categorical_columns:
            le = preprocessing.LabelEncoder()
            data[i] = le.fit_transform(data[i])
            self.categorical_encoders.append(le)
        
        #which columns are floats                            
        numerical_columns =  data.select_dtypes(include=[float])
        self.num_indices = [data.columns.get_loc(i) for i in numerical_columns]
        #which columns are ints
        integer_columns = data.select_dtypes(include=[int])
        self.int_indices = [data.columns.get_loc(i) for i in integer_columns]
        
        self.training_data = data
        #remove outcome from data since it is no longer necessary
        self.data = data.drop([outcome],axis=1).to_numpy()
        
        self.range = np.array([self.data.min(axis=0),self.data.max(axis=0)]).transpose()
        self.span = self.data.max(axis=0)-self.data.min(axis=0)
        

        self.no_weight = np.ones(len(self.column_names)-1)
        
        self.reset_feature_weights()
        
    def __getitem__(self,key):
        return self.data[key]
    
    def get_training_data(self):
        return self.training_data
    
    #transform a point from internal data format to original panda dataframe
    def backtransform_point(self,data):
        transformed_data = pd.DataFrame(data=data,columns= self.column_names[:-1])
        j=0
        for i in self.categorical_columns:
            le = self.categorical_encoders[j]
            transformed_data[i]=transformed_data[i].astype(int)
            transformed_data[i]= le.inverse_transform(transformed_data[i].values)
            j+=1
        return transformed_data
    
    #set feature weight of column
    def set_feature_weight(self,name,weight):
        idx = self.column_names.index(name)
        self.feature_weights[idx]=weight
    
    #reset all feature weights to inverse MAD
    def reset_feature_weights(self):
        norm_data = (self.data-self.range.T[0])/self.span
        self.feature_weights = scipy.stats.median_absolute_deviation(norm_data,axis=0,scale=1)
        
        #prevent infinity feature weight
        self.feature_weights[self.feature_weights==0]=1
        self.feature_weights[self.cat_indices] = 1
        #feature weight is inverse of mad. high weight=hard to change feature
        self.feature_weights=np.divide(1,self.feature_weights)

        
    def get_feature_weights(self):
        dic = {}
        for i in range(len(self.column_names)-1):
            dic[self.column_names[i]]=self.feature_weights[i]
        return dic
    
    #transform panda data point to internal data format
    def to_raw_format(self,x):
        x = x.copy()
        j=0
        for i in self.categorical_columns:
            le = self.categorical_encoders[j]
            x[i] = le.transform(x[i])
            j+=1
        return x.to_numpy()