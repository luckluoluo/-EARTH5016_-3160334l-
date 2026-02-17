%% ===================== Helmsdale Geothermal: Pure Conduction Map Only =====================
clear; clc; close all;

%% 1. 参数设置
target_dx = 50; target_dz = 50;        
W_core = 16000; W_buffer = 1000; 
Nx_core = round(W_core/target_dx); Nx_buffer = round(W_buffer/target_dx);
T_surface = 10; q_base = 0.050; h_conv = 15;
Stop_Tol = 1e-8; Max_Steps = 200000;

%% 2. 几何结构与属性
image_filename = 'units.tiff'; 
units_img = imread(image_filename);
if size(units_img,3)>1, units_img = units_img(:,:,1); end
Nz = round(size(units_img,1) * (Nx_core / size(units_img,2)));
units_core = double(imresize(units_img, [Nz, Nx_core], 'nearest'));
units = [repmat(units_core(:,1), 1, Nx_buffer), units_core];
dx = target_dx; dz = target_dz;

[K_map, Rho_map, Cp_map, Qr_map] = deal(zeros(Nz, Nx_core+Nx_buffer));
All_Props = [0.6, 1000, 4180, 0; 1.37, 2078, 1625, 1.0; 1.43, 2153, 1400, 1.0; 
             1.93, 2106, 1800, 1.0; 2.8, 2650, 850, 4.0; 2.5, 2700, 900, 1.0; 
             2.66, 2628, 836, 6.53; 2.7, 2299, 1031, 1.0; 1.7, 2000, 1500, 0.5; 
             2.2, 2400, 1000, 0.5];
for i = 1:10
    mask = (units == i);
    K_map(mask) = All_Props(i,1); Rho_map(mask) = All_Props(i,2);
    Cp_map(mask) = All_Props(i,3); Qr_map(mask) = All_Props(i,4)*1e-6;
end

%% 4. 初始化与迭代
dt = 0.5 * (min(dx,dz)^2 / (2 * max(K_map(:)./(Rho_map(:).*Cp_map(:)))));
T = ones(Nz, Nx_core+Nx_buffer) * T_surface;
for i = 1:Nz, T(i,:) = T_surface + (i*dz)*0.03; end 

for step = 1:Max_Steps
    T_old = T;
    d2T_dx2 = zeros(Nz, size(T,2)); d2T_dz2 = zeros(Nz, size(T,2));
    d2T_dx2(:, 2:end-1) = (T(:, 3:end) - 2*T(:, 2:end-1) + T(:, 1:end-2))/dx^2;
    d2T_dz2(2:end-1, :) = (T(3:end, :) - 2*T(2:end-1, :) + T(1:end-2, :))/dz^2;
    d2T_dx2(:, 1) = (2*T(:, 2) - 2*T(:, 1))/dx^2;
    d2T_dx2(:, end) = (2*T(:, end-1) - 2*T(:, end))/dx^2;
    d2T_dz2(end, :) = (2*T(end-1, :) - 2*T(end, :))/dz^2 + (2*q_base)./(K_map(end,:)*dz);
    
    dT_dt = (K_map.*(d2T_dx2+d2T_dz2) + Qr_map)./(Rho_map.*Cp_map);
    T = T + dt * dT_dt;
    T(1, :) = (h_conv*T_surface + (K_map(1,:)/dz).*T(2,:))./(h_conv + (K_map(1,:)/dz));
    T(units==1) = T_surface;
    
    if mod(step,2000)==0 && max(abs(T(:)-T_old(:))) < Stop_Tol, break; end
end

%% 8. 输出图像 (无标注纯净版)
T_plot = T(:, Nx_buffer+1:end);
[X, Z] = meshgrid((0:Nx_core-1)*dx/1000, (0:Nz-1)*dz/1000);

figure('Name','Pure Conduction Heat Field','Position', [100 100 1000 600]);
contourf(X, Z, T_plot, 50, 'LineStyle', 'none'); hold on;
colormap jet; cb = colorbar; ylabel(cb, 'Temperature (\circC)');

% 仅保留等温线
[C, h] = contour(X, Z, T_plot, [50 70 100 150], 'k', 'ShowText', 'on');
clabel(C, h, 'Color', 'w', 'FontSize', 10);

set(gca, 'YDir', 'reverse');
title('Pure Conduction Geothermal Model');
xlabel('Distance (km)'); ylabel('Depth (km)');
grid on;