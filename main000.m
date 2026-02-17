%% ===================== Helmsdale Geothermal: Explicit 2nd-Order Upwind =====================
% 特性：
% 1. 显式有限差分 (Explicit FDM)
% 2. 传导项：二阶中心差分
% 3. 平流项：二阶迎风格式 (Second-Order Upwind)
% 4. 稳定性：自动计算 CFL 步长限制

clear; clc; close all;

%% 1. 基础参数设置
target_dx = 50; target_dz = 50;        
W_core = 16000; W_buffer = 1000; 
T_surface = 10; q_base = 0.050; h_conv = 15;
Rho_f = 1000; Cp_f = 4180; Grad_P_drive = 500; 
Stop_Tol = 1e-6; Max_Steps = 100000; 

fprintf('正在加载几何并初始化二阶迎风模型...\n');

%% 2. 几何结构构建
image_filename = 'units.tiff'; 
if ~exist(image_filename, 'file'), error('未找到 units.tiff'); end
units_img = imread(image_filename);
if size(units_img, 3) > 1, units_img = units_img(:,:,1); end
Nx_core = round(W_core / target_dx); Nx_buffer = round(W_buffer / target_dx);
Nx = Nx_core + Nx_buffer; 
[img_h, img_w] = size(units_img);
Nz = round(img_h * (Nx_core / img_w)); 
units_core = double(imresize(units_img, [Nz, Nx_core], 'nearest'));
units = [repmat(units_core(:, 1), 1, Nx_buffer), units_core];
dx = target_dx; dz = target_dz;

%% 3. 物性参数映射
[K_map, Rho_map, Cp_map, Qr_map, KD_map] = deal(zeros(Nz, Nx));
% [kT, rho, cP, Qr, KD]
All_Props = [0.6, 1000, 4180, 0, 0; 1.37, 2078, 1625, 1.0, 9e-8; 1.43, 2153, 1400, 1.0, 3e-10; 
             1.93, 2106, 1800, 1.0, 1e-7; 2.8, 2650, 850, 4.0, 1e-11; 2.5, 2700, 900, 1.0, 1e-12; 
             2.66, 2628, 836, 6.53, 1e-11; 2.7, 2299, 1031, 1.0, 2e-7; 1.7, 2000, 1500, 0.5, 1e-9; 
             2.2, 2400, 1000, 0.5, 1e-11];
for i = 1:10
    mask = (units == i);
    K_map(mask) = All_Props(i, 1); Rho_map(mask) = All_Props(i, 2);
    Cp_map(mask) = All_Props(i, 3); Qr_map(mask) = All_Props(i, 4) * 1e-6;
    KD_map(mask) = All_Props(i, 5);
end

% 速度场与稳定性检查 (CFL Condition)
Vz_map = -1.0 * KD_map * Grad_P_drive; 
Vz_map(abs(Vz_map) > 1e-6) = -1e-6; 
alpha = K_map ./ (Rho_map .* Cp_map);
dt_limit = 0.45 * min([min(dx^2./(2*alpha(:))), min(dz^2./(2*alpha(:))), min(dz./(abs(Vz_map(:))+1e-15))]);
dt = dt_limit; 

%% 4. 初始化温度场 (带初始梯度的场)
T = zeros(Nz, Nx);
for i = 1:Nz
    T(i,:) = T_surface + (i-1)*dz * 0.025; % 25 C/km 初始梯度
end
T_old = T;

%% 5. 显式迭代主循环
fprintf('开始迭代... \n');
for step = 1:Max_Steps
    % --- A. 传导项计算 ---
    d2T_dx2 = zeros(Nz, Nx); d2T_dz2 = zeros(Nz, Nx);
    % 内部点
    d2T_dx2(:, 2:end-1) = (T(:, 3:end) - 2*T(:, 2:end-1) + T(:, 1:end-2)) / dx^2;
    d2T_dz2(2:end-1, :) = (T(3:end, :) - 2*T(2:end-1, :) + T(1:end-2, :)) / dz^2;
    % 边界处理
    d2T_dx2(:, 1) = (2*T(:, 2) - 2*T(:, 1)) / dx^2; % 绝热
    d2T_dx2(:, end) = (2*T(:, end-1) - 2*T(:, end)) / dx^2;
    d2T_dz2(end, :) = (2*T(end-1, :) - 2*T(end, :)) / dz^2 + (2*q_base) ./ (K_map(end,:) * dz); % 下边界热流
    
    % --- B. 平流项计算 (二阶迎风格式) ---
    % 当 Vz < 0 (向上流)，使用下游节点 (i, i+1, i+2)
    dT_dz_adv = zeros(Nz, Nx);
    % 1. 内部节点使用二阶格式
    dT_dz_adv(1:Nz-2, :) = (-3*T(1:Nz-2, :) + 4*T(2:Nz-1, :) - T(3:Nz, :)) / (2*dz);
    % 2. 靠近下边界的节点退化为一阶，避免越界 (修正了你遇到的维度错误)
    dT_dz_adv(Nz-1, :) = (T(Nz, :) - T(Nz-1, :)) / dz; 
    dT_dz_adv(Nz, :)   = (T(Nz, :) - T(Nz-1, :)) / dz; 

    Adv_term = -1 * (Rho_f * Cp_f) .* Vz_map .* dT_dz_adv;
    
    % --- C. 更新温度场 ---
    dT_dt = (K_map .* (d2T_dx2 + d2T_dz2) + Qr_map + Adv_term) ./ (Rho_map .* Cp_map);
    T = T + dt * dT_dt;
    
    % --- D. 上边界 Robin 条件 ---
    T(1, :) = (h_conv * T_surface + (K_map(1,:)/dz) .* T(2,:)) ./ (h_conv + (K_map(1,:)/dz));
    T(units == 1) = T_surface; % 湖水/地表水恒温
    
    % --- E. 收敛检查 ---
    if mod(step, 2000) == 0
        err = max(abs(T(:) - T_old(:)));
        if err < Stop_Tol, break; end
        T_old = T;
        if any(isnan(T(:))), error('数值发散！请检查参数或减小 dt'); end
        fprintf('步数: %6d | 残差: %.2e | 最高温: %.1f C\n', step, err, max(T(:)));
    end
end

%% 6. 绘图输出 (核心区 16km)
T_plot = T(:, Nx_buffer+1:end);
[X, Z] = meshgrid((0:Nx_core-1)*dx/1000, (0:Nz-1)*dz/1000);

figure('Name','Geothermal Model Results','Position', [100 100 1100 500]);
% 使用带有颜色的等高线填充图
[~, hf] = contourf(X, Z, T_plot, 40, 'LineStyle', 'none'); hold on;
colormap(jet(256));
c = colorbar; ylabel(c, 'Temperature (^\circC)', 'FontSize', 12);

% 专门绘制并标注 50, 70, 100, 150 度等温线
[C, h_iso] = contour(X, Z, T_plot, [50 70 100 150], 'k', 'ShowText', 'on', 'LineWidth', 1.2);
clabel(C, h_iso, 'FontSize', 10, 'Color', 'w', 'FontWeight', 'bold');

set(gca, 'YDir', 'reverse', 'FontSize', 12);
title('Scottish Highlands Geothermal Feasibility (2nd-Order Upwind)', 'FontSize', 14);
xlabel('Horizontal Distance (km)'); ylabel('Depth (km)');
grid on;

fprintf('计算完成。图像已成功生成。\n');