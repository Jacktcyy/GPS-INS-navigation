% ��ջ������� elm
clc
clear
close all
format compact
%% ��ȡ����
data=xlsread('wine.xls');%3����
input=data(:,1:end-1);
output=data(:,end);
%% ѡ����Լ���ѵ���� ���ѡ��100����Ϊѵ��Ѷ��   78����Ϊ��������
[m n]=sort(rand(1,178));
input_train=input(n(1:100),:)';
input_test=input(n(101:178),:)';
label_train=output(n(1:100),:)';
label_test=output(n(101:178),:)';
output_train=zeros(3,100);
output_test=zeros(3,78);
for i=1:100
    output_train(label_train(i),i)=1;
    
end
for i=1:78
output_test(label_test(i),i)=1;
end
%��һ��
[inputn_train,inputps]=mapminmax(input_train);
[inputn_test,inputtestps]=mapminmax('apply',input_test,inputps);
%% û���Ż���ELM
activation='sig';
TTYPE=1;
[IW,B,LW,TF,TYPE] = elmtrain(inputn_train,label_train,5,activation,1);
%% ELM�������
Tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
test_accuracy=(sum(label_test==Tn_sim))/length(label_test)
stem(label_test,'*')
hold on
plot(Tn_sim,'p')
title('û���Ż���ELM')
legend('�������','ʵ�����')
xlabel('������')
ylabel('����ǩ')
%% �ڵ����
inputnum=size(input_train,1);
hiddennum=5;
 %[bestchrom,trace]=gaforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%�Ŵ��㷨
%  [bestchrom,trace]=psoforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%����Ⱥ�㷨
% [bestchrom,trace]=batforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%�����㷨
%  [bestchrom,trace]=saforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%ģ���˻�
[bestchrom,trace]=antforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%��Ⱥ�㷨
% [bestchrom,trace]=afforelm(inputnum,hiddennum,TYPE,activation,inputn_train,label_train,inputn_test,label_test);%��Ⱥ�㷨

%% �Ż���������
 figure
[r c]=size(trace);
plot(trace,'b--');
title('��Ӧ������ͼ')
xlabel('��������');ylabel('�����ȷ��');
x=bestchrom;
%% �����ų�ʼ��ֵȨֵ����ELM����ѵ����Ԥ��
TYPE=1;
if TYPE  == 1
    T1  = ind2vec(label_train);
end
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum)';
%% train
W=reshape(w1,hiddennum,inputnum);
Q=size(inputn_train,2);
BiasMatrix = repmat(B1,1,Q);
tempH = W * inputn_train + BiasMatrix;
H = 1 ./ (1 + exp(-tempH));
LW = pinv(H') * T1';
%% test
T2=ind2vec(label_test);
Q=size(inputn_test,2);
BiasMatrix1 = repmat(B1,1,Q);
tempH1 = W * inputn_test + BiasMatrix1;
H1 = 1 ./ (1 + exp(-tempH1));
TY1=(H1'*LW)';
if TYPE  == 1
    temp_Y1=zeros(1,size(TY1,2));
for n=1:size(TY1,2)
    [max_Y,index]=max(TY1(:,n));
    temp_Y1(n)=index;
end
Y_train=temp_Y1;
end
youhua_test_accuracy=sum(Y_train==label_test)/length(label_test)
figure
stem(label_test,'*')
hold on
plot(Y_train,'p')
title('�Ż����ELM')
legend('�������','ʵ�����')