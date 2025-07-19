%% Initiation
SOCInit =.5;

%% EKF variable
Q  = 0.5;
R  = 0.7;
P0 = 0.7;
Ts = 0.1;
SOC0 =.5;

%% Varibales of Balancing system

%Variables of Resistancesbalancing block
PassiveBalancingRes(1) = 5;
PassiveBalancingRes(2) = 5;
PassiveBalancingRes(3) = 5;
PassiveBalancingRes(4) = 5;
PassiveBalancingRes(5) = 5;
PassiveBalancingRes(6) = 5;
PassiveBalancingRes(7) = 5;
PassiveBalancingRes(8) = 5;
PassiveBalancingRes(9) = 5;
PassiveBalancingRes(10) = 5;

balancingTime = 2000;
balancingRelaxationTime = 1000;


%Varibales related to Balancing in Chart
DeltaVTargetMin = 0.001;
balancingTime = 2000;
balancingRelaxationTime = 1000;

%% Variables of Contactors Block
%Contactor Resistance
ContactorChargerRes = 0.01;
ContactorInverterRes = 0.01;
ContactorCommonRes = 0.01;

%Relay Resistance
RelayPreChargerRes = 0.01;
RelayPreInverterRes = 0.01;

RelayDisChargerRes = 0.01;
RelayDisInverterRes = 0.01;

%Pre-charge Resistances
PreChargerRes = 1000;
PreInverterRes = 1000;
%Discharge Resistance
DisChargerRes = 250;
DisInverterRes = 250;

%FSM Constants
VbattThersholdChrg = 0.8; %[%]
VbattThresholdDis = 0.2;
VbattMin = 1; %[Volt]
DrivetrainEnDelay = 0.1; %[second]

%% Variables of Charger

ChargerRes = 0;
ChargerParRes = 10000;
ChargerCap = 5e-3;

%% Variables of Drivetrain

InverterRes = 0;
InverterParRes = 10000;
InverterCap = 2.5e-3;
%% Variable of Themperature

AmbientTemperature = 25; %Ambient temperature [degC]
CellThermalMass = 8e-5; %Thermal mass [J/K]
CellArea = 1e-100;
CellConvectiveCoeff = 0.1e-5; % Heat transfer coefficient [W/(m^2 * K)]

Cell2CellConductiveArea = 1e-2;
Cell2CellConductiveThickness = 0.1;
Cell2CellConductiveThermConduct = 401;

%% Safe Operation Area

% Temperature parameters
CellTemperatureLimitThreshold_Warning = 50 + 273.15; %[K]
CellTemperatureLimitThreshold_Fault = 65 + 273.15; %[K]

% Current parameters
CellCurrentLimitThreshold_Warning = 20; %[A]
CellCurrentLimitThreshold_Fault = 50; %[A]

% Voltage parameters
CellVoltageLimitHigh_Warning = 4.21; %[V]
CellVoltageLimitHigh_Fault = 4.25; %[V]

CellVoltageLimitLow_Warning = 2.70; %[V]
CellVoltageLimitLow_Fault = 2.65; %[V]

%% Run Simulink model

run("BMSModel.slx")