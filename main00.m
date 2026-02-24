%% ========================================================================
% HELMSDALE GEOTHERMAL: Buffered Optimized Model & Pathway Analysis
% Focuses on steady-state heat distribution and identifying 
% optimal drilling targets based on temperature thresholds.
% ========================================================================

clear; clc; close all;

%% 1. Model Configuration
target_dx = 50;  
target_dz = 50;        
W_core = 16000;         % Core display width (16km)
W_buffer = 1000;        % Left buffer to minimize boundary effects
W_total = W_core + W_buffer; 
T_surface = 10;        
Stop_Tol = 1e-6;       
Max_Steps = 100000;    
Rho_f = 1000; Cp_f = 4180; k_f = 0.6; q_base = 0.050;
h_conv = 15;            % Convection coefficient for surface
Grad_P_drive = 500; 
Total_Target_Time = 2e6; 
dt_years = 5;            
dt = dt_years * 365.25 * 24 * 3600; 

fprintf('Initializing buffered fluid-thermal coupled model...\n');

%% 2. Geometry Loading & Domain Expansion
image_filename = 'units.tiff'; 
if exist(image_filename, 'file')
    units_img = imread(image_filename);
    if size(units_img, 3) > 1, units_img = units_img(:,:,1); end
else
    error('Missing units.tiff');
end

% Mesh dimensioning
Nx_core = round(W_core / target_dx); 
Nx_buffer = round(W_buffer / target_dx);
Nx = Nx_core + Nx_buffer; 
[img_h, img_w] = size(units_img);
Nz = round(img_h * (Nx_core / img_w)); 

% Build total domain with left-side buffer
units_core = double(imresize(units_img, [Nz, Nx_core], 'nearest'));
left_edge_column = units_core(:, 1);
units_buffer = repmat(left_edge_column, 1, Nx_buffer);
units = [units_buffer, units_core]; 
H_model = Nz * target_dz;
dx = target_dx; dz = target_dz;

fprintf('Calculation width: %.2f km, Display width: %.2f km\n', W_total/1000, W_core/1000);

%% 3. Physical Property Assignment
[K_map, Rho_map, Cp_map, Qr_map, KD_map] = deal(zeros(Nz, Nx));
All_Props = [0.6, 1000, 4180, 0, 0;       % Unit 1
             1.37, 2078, 1625, 1.0, 9e-8; % Unit 2
             1.43, 2153, 1400, 1.0, 3.9e-10; 
             1.93, 2106, 1800, 1.0, 1.35e-7; 
             2.8, 2650, 850, 4.0, 1e-10;  
             2.5, 2700, 900, 1.0, 1e-11;  
             2.66, 2628, 836, 6.53, 1e-10; 
             2.7, 2299, 1031, 1.0, 2.16e-7; 
             1.7, 2000, 1500, 0.5, 1e-9;  
             2.2, 2400, 1000, 0.5, 1e-11];
for i = 1:10
    mask = (units == i);
    K_map(mask) = All_Props(i, 1);
    Rho_map(mask) = All_Props(i, 2);
    Cp_map(mask) = All_Props(i, 3);
    Qr_map(mask) = All_Props(i, 4) * 1e-6;
    KD_map(mask) = All_Props(i, 5);
end

%% 4. Velocity Field & Stability Check
Vz_map = zeros(Nz, Nx);
Vz_map(units == 8) = -1.0 * KD_map(units == 8) * Grad_P_drive;
max_v = 1e-6; Vz_map(abs(Vz_map) > max_v) = -max_v; % Prevent runaway velocities

alpha_max = max(K_map(:) ./ (Rho_map(:) .* Cp_map(:)));
dt = 0.8 * min(min(dx,dz)^2 / (2 * alpha_max), dz/(max(abs(Vz_map(:)))+1e-12));

%% 6. Field Initialization
T = ones(Nz, Nx) * T_surface;
for i = 1:Nz, T(i,:) = T_surface + (i*dz) * 0.025; end
T_old = T;

%% 7. Iterative Solver
fprintf('Executing simulation loop...\n');
for step = 1:Max_Steps
    d2T_dx2 = zeros(Nz, Nx); d2T_dz2 = zeros(Nz, Nx);
    d2T_dx2(:, 2:end-1) = (T(:, 3:end) - 2*T(:, 2:end-1) + T(:, 1:end-2)) / dx^2;
    d2T_dz2(2:end-1, :) = (T(3:end, :) - 2*T(2:end-1, :) + T(1:end-2, :)) / dz^2;
    
    % BCs: No-flux sides, Basal heat flux bottom
    d2T_dx2(:, 1) = (2*T(:, 2) - 2*T(:, 1)) / dx^2;
    d2T_dx2(:, end) = (2*T(:, end-1) - 2*T(:, end)) / dx^2;
    term_flux = (2 * q_base) ./ (K_map(end, :) * dz);
    d2T_dz2(end, :) = (2*T(end-1, :) - 2*T(end, :)) / dz^2 + term_flux;
    
    % Upwind Advection
    dT_dz_adv = zeros(Nz, Nx);
    up_flow = (Vz_map < 0);
    if any(up_flow(:))
        T_pad_down = [T(2:end, :); T(end, :) + (T(end,:)-T(end-1,:))]; 
        dT_dz_adv(up_flow) = (T_pad_down(up_flow) - T(up_flow)) / dz;
    end
    
    Adv_term = -1 * (Rho_f * Cp_f) .* Vz_map .* dT_dz_adv;
    dT_dt = (K_map .* (d2T_dx2 + d2T_dz2) + Qr_map + Adv_term) ./ (Rho_map .* Cp_map);
    T_new = T_old + dt * dT_dt;
    
    % Surface BC (Robin/Convection)
    k_surf = K_map(1, :);
    T_new(1, :) = (h_conv * T_surface + (k_surf / dz) .* T_new(2, :)) ./ (h_conv + (k_surf / dz));
    T_new(units == 1) = T_surface; 
    
    if max(abs(T_new(:)-T(:))) < Stop_Tol, break; end
    T = T_new; T_old = T;
end

%% 8. Results Plotting & Pathway Mapping
T_plot = T(:, Nx_buffer+1:end);
units_plot = units(:, Nx_buffer+1:end);
[X_plot, Z_plot] = meshgrid((0:Nx_core-1)*dx/1000, (0:Nz-1)*dz/1000);

figure('Name','Final Optimized Model','Position', [100 100 1000 600]);
contourf(X_plot, Z_plot, T_plot, 50, 'LineStyle', 'none'); hold on;
colormap jet; cb = colorbar; ylabel(cb, 'Temperature (\circC)');

% Overlay Fault Zone mask (Unit 8)
fz_mask = (units_plot == 8);
h_fz = imagesc([0 W_core/1000], [0 H_model/1000], fz_mask);
set(h_fz, 'AlphaData', fz_mask * 0.3);

% Locate the shallowest 70C point for drilling efficiency
depth_70 = zeros(1, Nx_core);
for ix = 1:Nx_core
    [~, iz] = min(abs(T_plot(:, ix) - 70));
    depth_70(ix) = iz * dz / 1000;
end
[~, best_x_idx] = min(depth_70(5:end-5)); 
best_x_idx = best_x_idx + 4;
start_x_km = (best_x_idx - 1) * dx / 1000;

% Point of interest: 70C Isotherm intersection (p_70)
[~, iz_70] = min(abs(T_plot(:, best_x_idx) - 70));
p_70 = [start_x_km, (iz_70-1)*dz/1000];

% Projected end-of-well point in Fault Zone (p_end)
[fz_z, fz_x] = find(units_plot == 8);
p_fit = polyfit(fz_z*dz/1000, fz_x*dx/1000, 1);
p_end = [polyval(p_fit, 2.8), 2.8];

% Engineering fixed reference point (p_fixed)
p_fixed = [13.65, 1.8];

% Visual annotations
plot([start_x_km, start_x_km, p_end(1)], [0, 2.0, 2.8], 'w-s', 'LineWidth', 3, 'MarkerSize', 6, 'MarkerFaceColor', 'r'); % Primary Path
plot([p_fixed(1), p_70(1)], [p_fixed(2), p_70(2)], 'y--', 'LineWidth', 2); % Connection A
plot([p_fixed(1), p_end(1)], [p_fixed(2), p_end(2)], 'c--', 'LineWidth', 2); % Connection B
plot(p_fixed(1), p_fixed(2), 'p', 'MarkerSize', 12, 'MarkerFaceColor', 'm', 'MarkerEdgeColor', 'w'); % Fixed Target
plot(p_70(1), p_70(2), 'go', 'MarkerSize', 8, 'LineWidth', 2); % Intersection point

% Isotherms
[C, h] = contour(X_plot, Z_plot, T_plot, [50 70 100 150], 'k', 'ShowText', 'on');
clabel(C, h, 'Color', 'w', 'FontWeight', 'bold');

set(gca, 'YDir', 'reverse');
title(sprintf('Target: 70^oC Intersect at %.2f km | Fixed Point (13.65, 1.8)', p_70(2)));
xlabel('Distance (km)'); ylabel('Depth (km)');
grid on;