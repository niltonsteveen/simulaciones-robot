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

Tipo=input('Ingrese según lo desee: \n 1. Funciones Discriminantes Gausianas\n 2. K-nn\n 3. RNA\n 4. Random Forest \n 5. Máquinas de soporte Vectorial\n input: ');

if Tipo==1 %Funciones discriminantes gaussianas
    TipoValidacion=input('Ingrese 1 para validacion Bootstrap ó 2 para validacion cruzada\n input:');
    Rept=10;
    EficienciaTest=zeros(1,Rept);
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
    elseif TipoValidacion==2
        for fold=1:Rept
            %%% Se hace la partición de las muestras %%%
            %%%      de entrenamiento y prueba       %%%

            rng('default');
            particion=cvpartition(N,'Kfold',Rept);
            indices=particion.training(fold);
            Xtrain=X(particion.training(fold),:);
            Xtest=X(particion.test(fold),:);
            Ytrain=Y(particion.training(fold));
            Ytest=Y(particion.test(fold));

            %%% Normalización %%%

            [Xtrain,mu,sigma]=zscore(Xtrain);
            Xtest=normalizar(Xtest,mu,sigma);

            %%%%%%%%%%%%%%%%%%%%
            Yest=classify(Xtest,Xtrain,Ytrain,'mahalanobis');
            %%% Se encuentra la eficiencia y el error de clasificación %%%

            MatrizConfusion = zeros(NumClases,NumClases);
            for i=1:size(Xtest,1)
                MatrizConfusion(Yest(i),Ytest(i)) = MatrizConfusion(Yest(i),Ytest(i)) + 1;
            end
            diagonal = diag(MatrizConfusion);
            EficienciaTest(fold) = sum(diag(MatrizConfusion))/sum(sum(MatrizConfusion));
        end
        Eficiencia = mean(EficienciaTest);
        Texto=['La eficiencia obtenida fue = ', num2str(Eficiencia)];
        disp(Texto);
    end
end