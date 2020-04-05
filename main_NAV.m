addpath('quaternion_library');      % include quaternion library
close all;                          % close all figures
clear;                              % clear all variables
clc;                                % clear the command terminal
%% Import and plot sensor data
%Data = importdata('NAV_2.mat');
Data = xlsread('dat.xlsx');


train_data=xlsread('train data .xlsx');
test_data=xlsread('test data .xlsx');
train_input=train_data(:,5:end-3);
train_output=train_data(:,end-3:end);
%test_input=test_data(:,1:end-3);
%test_output=test_data(:,end-3:end);
k=rand(1,2000);
[m,n]=sort(k);
input_train=train_input(n(1:1900),:)';
output_train=train_output(n(1:1900));     %һ��
%input_test=test_input(n(1901:2000),:)';
%output_test=test_output(n(1901:2000));

[inputn,inputps]=mapminmax(input_train);          %�����һ��
[outputn,outputps]=mapminmax(output_train);    %�����һ��

net=newff(inputn,outputn,5);
net.trainParam.epochs=100;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00004;

net=train(net,inputn,outputn);


TF = 'sig';
TYPE = 0;
[IW,B,LW,TF,TYPE] = elmtrain(inputn,outputn,30);


gx=Data(:,27); %%����������
gy=Data(:,28); 
gz=Data(:,29); 
ax=Data(:,9); %%���ٶȼ�����
ay=Data(:,10); 
az=Data(:,11); 
mx=Data(:,15); %%��ǿ������
my=Data(:,16); 
mz=Data(:,17); 
rad2deg=180/pi;
deg2rad=pi/180;
time(1)=0;
quaternion = zeros(length(time), 4);
%% Process sensor data through algorithm
Pitch0=asin(ax(1)/sqrt(ax(1)*ax(1)+ay(1)*ay(1)+az(1)*az(1)));%��ʼ�Ǽ���
Roll0=atan2(-ay(1),-az(1));
Yaw0=atan2(-my(1)*cos(Roll0)+mz(1)*sin(Roll0),mx(1)*cos(Pitch0)+my(1)*sin(Pitch0)*sin(Roll0)+mz(1)*sin(Pitch0)*cos(Roll0))-8.3*pi/180;
[q0,q1,q2,q3]=Quaternion_FromEulerAngle(Roll0,Pitch0,Yaw0);%��ʼ��Ԫ������
q(1,:)=[q0,q1,q2,q3];
q_1(1,:)=[q0,q1,q2,q3];
Pitch(1)= Pitch0*rad2deg; 
Roll(1) =Roll0*rad2deg;
Yaw(1) = Yaw0*rad2deg;
tt(1)=0;                                      %�㷨���η���ʱ�����
SamplePeriod=0.02;                            %���ݲ���ʱ��
Beta=0.009;                                    %�ݶ��½����ݶȲ�������
Kp=2;Ki=0.1;                                  %�������������ϵ������
eInt(1,:)=[0 0 0];                            %���������ֵ
Length= size(Data,1);                        
Re=6378137;r=6356752.3142;f=1/298.257;e=0.0818;wie=7.292e-5;%Re������ r�̰��� f������� e����ƫ���� wie������ת������
Ven(:,1)=[Data(1,6) Data(1,7) Data(1,8)]';
E(1)=Data(1,2)*deg2rad;L(1)=Data(1,3)*deg2rad;H(1)=Data(1,4);
fn(:,1)=[0 0 0]';g(:,1)=[0 0 0]';Venq1(:,1)=[0 0 0]'; Venq2(:,1)=[0 0 0]';
g0=9.7803;
%�ߵ���ص�����ͳ������
Q_wg  = (1/(57*3600))^2;         %���ݵ����Ư��Ϊ0.5��ÿСʱ
Q_wa  = ((0.5e-4)*g0)^2;                  %���ٶȼƵ����ƫ��Ϊ0.5e-4*g  
Q 		= diag([Q_wg Q_wg Q_wg,  Q_wa Q_wa Q_wa]);%ϵͳ��������
%Q=diag([1e-9 1e-9 1e-9 1e-7 1e-7 1e-7 ]);

Tg 		= 300*ones(3,1);                          %Tg     ���������Ư�����ʱ��
Ta 		= 1000*ones(3,1);                         %Ta     �ӱ����Ư�����ʱ��
%error_cz=0.01*pi/180/3600;%%%�������Ư�� 0.5��/ʱ��
%tg=300; %%%���ʱ��tg=300s
%kt=sqrt(2*error_cz^2/tg);%%%��������ǿ�Ⱦ�����
%ta=1000;%%���ʱ��
%Ra=1.0e-4*g0;%%������Ra=0.0001*g
%ka=sqrt(2*Ra^2/ta);%%%%��������ǿ�Ⱦ�����
%zt=0.05*deg2rad*ones(1,3);  %%%%���ֵ��̬����������������� 
%wz=[0.001/60*pi/180 0.001/60*pi/180 0.1];   %%%%���ֵλ���������������� 
%sd=0.0001*[1,1,1];    %%%%%���ֵ�ٶ��������������� 
%fc=[zt sd wz kt kt kt ka ka ka ].^2;%%%%%�µķ�����Խ���Ԫ��
%Q=diag(fc);
%R = 1e-6*eye(6);                               %GPS��������������
Rlamt=1e-5*pi/(60*180); %%%��γ����������,������
Rl=1e-5*pi/(60*180);
Rh=1e-11; %%%�߶���������,��λ ��
Rvx=1e-7; %%%�ٶ���������,��λ ��/��
Rvy=1e-7;
Rvz=5e-9;
K=[Rlamt Rl Rh Rvx Rvy Rvz];%
R=diag(K);
m=0;
PP0(1:15,1:15,1) = diag([0.1/(57) 0.1/(57) 0.1/57, 0.01 0.01 0.01, 0 0 0, 0.1/(57*3600) 0.1/(57*3600) 0.1/(57*3600), (1e-4)*g0 (1e-4)*g0 (1e-4)*g0].^2);   %��ʼ���Э������,���ٶȼƵĳ�ʼƫֵ��ȡ1e-4*g ���ݵĳ�ֵƯ��ȡ0.1��ÿСʱ
X(:,:,1)= zeros(15,1);                            %��ʼ״̬
 a=0;                               %tao    ��������
E_pv2(1,:)=[0 0 0 0 0 0];



for t = 2:Length
    tic;
    q(t,:) = madgwickAHRS(Data(t,27:29), Data(t,9:11), Data(t,15:17),q(t-1, :),Beta,SamplePeriod);	%�ݶ��½���  
    [q_1(t,:),eInt(t,:)] = MahonyAHRSupdate( Data(t,27:29), Data(t,9:11), Data(t,15:17),q_1(t-1, :),SamplePeriod,Ki,Kp,eInt(t-1,:));%Mahony��
    tt(t)=toc;%�����㷨����ʱ��
    
    T=[ 1 - 2 * (q(t,4) *q(t,4) + q(t,3) * q(t,3)) 2 * (q(t,2) * q(t,3) +q(t,1) * q(t,4)) 2 * (q(t,2) * q(t,4)-q(t,1) * q(t,3));
        2 * (q(t,2) * q(t,3)-q(t,1) * q(t,4)) 1 - 2 * (q(t,4) *q(t,4) + q(t,2) * q(t,2)) 2 * (q(t,3) * q(t,4)+q(t,1) * q(t,2));
        2 * (q(t,2) * q(t,4) +q(t,1) * q(t,3)) 2 * (q(t,3) * q(t,4)-q(t,1) * q(t,2)) 1 - 2 * (q(t,2) *q(t,2) + q(t,3) * q(t,3))];%cnb
    Roll(t)  = atan2(T(2,3),T(3,3))*rad2deg;
    Pitch(t) = asin(-T(1,3))*rad2deg;
    Yaw(t)   = atan2(T(1,2),T(1,1))*rad2deg-8.3;
	%%Mahony��������̬
	%[q1(t,:),eInt(t,:)] = MahonyAHRSupdate( Data(t,27:29), Data(t,9:11), Data(t,15:17),q(t-1, :),SamplePeriod,Ki,Kp,eInt(t-1,:));%Mahony��
	
	T1=[ 1 - 2 * (q_1(t,4) *q_1(t,4) + q_1(t,3) * q_1(t,3)) 2 * (q_1(t,2) * q_1(t,3) +q_1(t,1) * q_1(t,4)) 2 * (q_1(t,2) * q_1(t,4)-q_1(t,1) * q_1(t,3));
        2 * (q_1(t,2) * q_1(t,3)-q_1(t,1) * q_1(t,4)) 1 - 2 * (q_1(t,4) *q_1(t,4) + q_1(t,2) * q_1(t,2)) 2 * (q_1(t,3) * q_1(t,4)+q_1(t,1) * q_1(t,2));
        2 * (q_1(t,2) * q_1(t,4) +q_1(t,1) * q_1(t,3)) 2 * (q_1(t,3) * q_1(t,4)-q_1(t,1) * q_1(t,2)) 1 - 2 * (q_1(t,2) *q_1(t,2) + q_1(t,3) * q_1(t,3))];%cnb
    Roll1(t)  = atan2(T1(2,3),T1(3,3))*rad2deg;
    Pitch1(t) = asin(-T1(1,3))*rad2deg;
    Yaw1(t)   = atan2(T1(1,2),T1(1,1))*rad2deg-8.3;
    
    %Rm(t-1)=Re*(1-e*e)/power((1-e*e*sin(L(t-1))*sin(L(t-1))),1.5);%�������ʰ뾶
   % Rn(t-1)=Re/sqrt(1-e*e*sin(L(t-1))*sin(L(t-1)));               %î�����ʰ뾶
    Rm(t-1)=Re*(1-2*f+3*f*sin(L(t-1))*sin(L(t-1))); 
    Rn(t-1)=Re*(1+f*sin(L(t-1))*sin(L(t-1)));
    R0=sqrt(Rm(t-1)*Rn(t-1));                                     %����ƽ���뾶
    Wien(t-1,:)=[wie*cos(L(t-1)) 0 -wie*sin(L(t-1))];             %����������ϵ�е�����ת���ٶ�
    Wenn(t-1,:)=[Ven(2,t-1)/(Rn(t-1)+H(t-1)) -Ven(1,t-1)/(Rm(t-1)+H(t-1)) -Ven(2,t-1)*tan(L(t-1))/(Rm(t-1)+H(t-1))];%��������ϵ����ڵ����������ϵ��ת��������
    g0=9.780318;%*(1+5.3024*1e-3*sin(L(t-1))*sin(L(t-1))-5.9*1e-6*sin(2*L(t-1))*sin(2*L(t-1)));
    %g(:,t)=[0 0 g0/(1+H(t-1)/R0)^2]';
    %g(:,t)=[0 0 g0*(1-2*H(t-1)/Re)]';
    Fn(:,t-1)=T'*([Data(t-1,9:11)]')*9.8;                         %�ֽ⵽��������ϵ�еı���
    WWX=[0 2*Wien(t-1,3)+Wenn(t-1,3) -(2*Wien(t-1,2)+Wenn(t-1,2));-(2*Wien(t-1,3)+Wenn(t-1,3)) 0 2*Wien(t-1,1)+Wenn(t-1,1);2*Wien(t-1,2)+Wenn(t-1,2) -(2*Wien(t-1,1)+Wenn(t-1,1)) 0];%���������
    Venq1(:,t)=Fn(:,t-1)+WWX*Ven(:,t-1)+[0 0 10.05]';                    %�ٶ�΢��
    Venq2(:,t)=Data(t-1,24:26)';
   
    Ven(:,t)=Ven(:,t-1)+Venq1(:,t)*SamplePeriod;    
    L(t)=L(t-1)+(Ven(1,t-1)/(Rm(t-1)+H(t-1)))*SamplePeriod;              %γ��
    E(t)=E(t-1)+(Ven(2,t-1)/(cos(L(t-1))*(Rn(t-1)+H(t-1))))*SamplePeriod;%����
    H(t)=H(t-1)-Ven(3,t-1)*SamplePeriod;                                 %�߶�
    time(t)=time(t-1)+0.02;
    
    input_test=test_data(t,1:end-3);
    test_output=test_data(t,end-3:end);
    inputn_test=mapminmax('apply',input_test,inputps); %����inputps��ʽ���й�һ��
    an=sim(net,inputn_test);
    BPoutput(t,:)=mapminmax('reverse',an,outputps);%����outputps��ʽ���з���һ���������ʵֵ
    tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
    T_sim(t,:)=mapminmax('reverse',tn_sim,outputps);
    %error(:,t)=test_output(:,t)-T_sim(:,t);
    m=m+1;
    if Data(t,36)==1
        a=a+1;
        tao=m*0.02;
        m=0;
        Dpv=[L(t)*rad2deg-Data(t,3),E(t)*rad2deg-Data(t,2),H(t)-Data(t,4),Ven(1,t)-Data(t,6),Ven(2,t)-Data(t,7),Ven(3,t)-Data(t,8)];       %�ߵ���Gps����λ���ٶ�֮��
        [E_v, E_p,PP,XX] = kalman_GPS_INS_pv(Dpv, Ven(:,t), L(t),H(t), T, Fn(:,t-1), Q, R, Tg, Ta, tao,Rm(t-1),Rn(t-1),PP0(:,:,a),X(:,:,a)) ;%GPS/INSλ���ٶ���� �������˲�
        PP0(:,:,a+1)=PP;
        X(:,:,a+1)=XX;
        %{
        if t~=Length
        Data(t+1,9:11)=Data(t+1,9:11)-X(13:15,a+1)';
        Data(t+1,27:29)=Data(t+1,27:29)-X(10:12,a+1)';
        end
        %}
        % Q=Q1;
        Ven(:,t)=Ven(:,t)-[E_v(1); E_v(2); 0.2*E_v(3)];
        L(t)=L(t)-0.29*E_p(1);
        E(t)=E(t)-0.32*E_p(2);
        H(t)=H(t)-E_p(3);
   end   
  % E_pv2(t,:)=[Ven(1,t)-Data(t,21),Ven(2,t)-Data(t,22),Ven(3,t)-Data(t,23),E(t)*rad2deg-Data(t,18),L(t)*rad2deg-Data(t,19),H(t)-Data(t,20)]; 
end
%save E_pv2;
%save Ven;save L;save E;save H;
for i=1:Length
   L(i)=L(i)*rad2deg; 
    E(i)=E(i)*rad2deg; 
end
%figure('Name', 'Euler Angles');
%hold on;
%plot(time, Roll, 'r',time,Data(:,30)*57.3,'m');
%plot(time, Pitch, 'g',time,Data(:,31)*57.3,'c');
%plot(time, Yaw, 'b',time,Data(:,32)*57.3,'k');
%title('Euler angles');
%xlabel('Time (s)');
%ylabel('Angle (deg)');
%legend('Roll����ֵ','Roll�ο�ֵ', 'Pitch����ֵ', 'Pitch�ο�ֵ','Yaw����ֵ','Yaw�ο�ֵ');
%hold off;

figure('Name', 'Roll1');
hold on;
plot(time, Roll, 'r',time,Data(:,30)*57.3,'m');
plot(time, Roll1,'g',time,Data(:,30)*57.3,'m');
title('Roll1 angles');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('Roll�ݶ��㷨����ֵ','Roll�ο�ֵ', 'Roll Mahony������ֵ');
hold off;

figure('Name', 'Pitch');
hold on;
plot(time, Pitch, 'r',time,Data(:,31)*57.3,'m');
plot(time, Pitch1, 'g' ,time,Data(:,31)*57.3,'m');
title('Pitch angles');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('Pitch�ݶ��㷨����ֵ','Pitch�ο�ֵ', 'Pitch Mahony������ֵ');
hold off;

figure('Name', 'Yaw');
hold on;
plot(time, Yaw, 'r',time,Data(:,32)*57.3,'m');
plot(time, Yaw1, 'g' ,time,Data(:,32)*57.3,'m');
title('Yaw angles');
xlabel('Time (s)');
ylabel('Angle (deg)');
legend('Yaw�ݶ��㷨����ֵ','Yaw�ο�ֵ', 'Yaw Mahony������ֵ');
hold off;

figure('Name', '1');
hold on;
plot(time,E, 'r');
plot(time, Data(:,18), 'g');
title('���ȶԱ�ͼ','fontsize',20);
xlabel('Time (s)','fontsize',16);
ylabel(' longitude','fontsize',16);
d=legend('���ȼ���ֵ', '���Ȳο�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
figure('Name', '2');
hold on;
plot(time, L, 'r');
plot(time, Data(:,19), 'g');
title('γ�ȶԱ�ͼ','fontsize',20);
xlabel('Time (s)','fontsize',16);
ylabel('latitude','fontsize',16);
d=legend('γ�ȼ���ֵ', 'γ�Ȳο�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
figure('Name', '3');
hold on;
plot(time, H, 'r');
plot(time, Data(:,20), 'g');
title('�߶ȶԱ�ͼ','fontsize',20);
xlabel('Time (s)','fontsize',16);
ylabel(' H','fontsize',16);
d=legend('�߶ȼ���ֵ', '�߶Ȳο�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
figure('Name', '4');
hold on;
plot(time, Ven(1,:), 'r');
plot(time, Data(:,21), 'g');
plot(time,T_sim(:,1), 'b');
title('�����ٶȶԱ�ͼ','fontsize',20);
xlabel('Time (s)','fontsize',16);
ylabel('�����ٶ�','fontsize',16);
d=legend('�����ٶȼ���ֵ', '�����ٶȲο�ֵ','elm�����ٶ�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
figure('Name', '5');
hold on;
plot(time,Ven(2,:), 'r');
plot(time, Data(:,22), 'g');
plot(time, T_sim(:,2), 'b');
title('�����ٶȶԱ�ͼ','fontsize',20);
xlabel('Time (s)','fontsize',16);
ylabel('�����ٶ�','fontsize',16);
d=legend('�����ٶȼ���ֵ', '�����ٶȲο�ֵ','elm�����ٶ�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
figure('Name', '6');
hold on;
plot(time, Ven(3,:), 'r');
plot(time, Data(:,23), 'g');
plot(time, T_sim(:,3), 'b');
title('�����ٶȶԱ�ͼ','fontsize',20);
xlabel('Time(s)','fontsize',16);
ylabel('�����ٶ�','fontsize',16);
d=legend('�����ٶȼ���ֵ', '�����ٶȲο�ֵ','elm�����ٶ�ֵ');
set(d,'Fontsize',10,'LineWidth',2,'box','off');
hold off;
