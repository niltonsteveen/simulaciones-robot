clc
clear all
% close all
load('sensor_data.csv');

%Se cargan las muestras
X=sensor_data(1:end,1:end-1);

%Se cargan las salidas
Y=sensor_data(1:end,end);

N=size(X,1); %Numero de muestras
NClases=length(unique(Y));

Tipo=input('Ingrese seg�n lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. M�quinas de soporte Vectorial\n input: ');

if Tipo==1 %Funciones discriminantes gaussianas
    TipoValidacion=input('Ingrese 1 para validacion Bootstrap � 2 para validacion cruzada');
    if TipoValidacion==1
        ind=randperm(N);
        Xtrain=X(ind(1:3819),:);
        Xtest=X(ind(3820:end),:);
        Ytrain=Y(ind(1:3819),:);
        Ytest=Y(ind(3820:end),:);
        
        [Xtrain,mu,sigma]=zscore(Xtrain);
        Xtest=normalizar(Xtest,mu,sigma);
        
        Yest=classify(Xtest,Xtrain,Ytrain,'mahalanobis');
        MatrizConfusion = zeros(NClases,NClases);
        
        for i=1:size(Xtest,1)
            MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
        end
        diagonal = diag(MatrizConfusion);
        Eficiencia = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
    end
end