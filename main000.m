%% ========================================================================
% HELMSDALE GEOTHERMAL: Explicit 2nd-Order Upwind Solver
% This script simulates heat transport using an explicit Finite Difference 
% Method (FDM) featuring a 2nd-order upwind scheme for advection to 
% minimize numerical diffusion.
% ========================================================================

clear; clc; close all;

%% 1. Model Configuration & Parameters
target_dx = 50; target_dz = 50;        % Target grid resolution (meters)
W_core = 16000; W_buffer = 1000;       % Domain widths: 16km study area + 1km buffer
T_surface = 10; q_base = 0.050; h_conv = 15; % Boundary conditions (Surface Temp, Basal Heat Flow, Convection)
Rho_f = 1000; Cp_f = 4180; Grad_P_drive = 500; % Fluid properties & pressure gradient
Stop_Tol = 1e-6; Max_Steps = 100000;   % Iteration controls

fprintf('Loading geometry and initializing 2nd-order upwind model...\n');

%% 2. Domain Geometry Construction
image_filename = 'units.tiff'; 
if ~exist(image_filename, 'file'), error('Missing geometry file: units.tiff'); end
units_img = imread(image_filename);
if size(units_img, 3) > 1, units_img = units_img(:,:,1); end

% Calculate grid dimensions based on target resolution
Nx_core = round(W_core / target_dx); Nx_buffer = round(W_buffer / target_dx);
Nx = Nx_core + Nx_buffer; 
[img_h, img_w] = size(units_img);
Nz = round(img_h * (Nx_core / img_w)); 

% Resize unit map and append the left-side buffer (repeating the edge column)
units_core = double(imresize(units_img, [Nz, Nx_core], 'nearest'));
units = [repmat(units_core(:, 1), 1, Nx_buffer), units_core];
dx = target_dx; dz = target_dz;

%% 3. Physical Property Mapping
[K_map, Rho_map, Cp_map, Qr_map, KD_map] = deal(zeros(Nz, Nx));
% Lookup Table: [Thermal Cond (kT), Density (rho), Heat Cap (cP), Radiogenic Heat (Qr), Permeability (KD)]
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

% Darcy Velocity and Stability (CFL) Check
Vz_map = -1.0 * KD_map * Grad_P_drive; 
Vz_map(abs(Vz_map) > 1e-6) = -1e-6; % Velocity cap for stability
alpha = K_map ./ (Rho_map .* Cp_map); % Thermal diffusivity
dt_limit = 0.45 * min([min(dx^2./(2*alpha(:))), min(dz^2./(2*alpha(:))), min(dz./(abs(Vz_map(:))+1e-15))]);
dt = dt_limit; 

%% 4. Initial Temperature Field (Geothermal Gradient)
T = zeros(Nz, Nx);
for i = 1:Nz
    T(i,:) = T_surface + (i-1)*dz * 0.025; % Assume 25 C/km starting gradient
end
T_old = T;

%% 5. Main Explicit Iteration Loop
fprintf('Starting solver loop... \n');
for step = 1:Max_Steps
    % --- A. Conduction Terms (2nd-Order Central Difference) ---
    d2T_dx2 = zeros(Nz, Nx); d2T_dz2 = zeros(Nz, Nx);
    % Interior points
    d2T_dx2(:, 2:end-1) = (T(:, 3:end) - 2*T(:, 2:end-1) + T(:, 1:end-2)) / dx^2;
    d2T_dz2(2:end-1, :) = (T(3:end, :) - 2*T(2:end-1, :) + T(1:end-2, :)) / dz^2;
    % Boundary Neumann/Insulation treatment
    d2T_dx2(:, 1) = (2*T(:, 2) - 2*T(:, 1)) / dx^2; % Adiabatic left
    d2T_dx2(:, end) = (2*T(:, end-1) - 2*T(:, end)) / dx^2; % Adiabatic right
    d2T_dz2(end, :) = (2*T(end-1, :) - 2*T(end, :)) / dz^2 + (2*q_base) ./ (K_map(end,:) * dz); % Basal heat flux
    
    % --- B. Advection Terms (2nd-Order Upwind) ---
    % Since Vz < 0 (upward flow), we use downstream nodes (i, i+1, i+2)
    dT_dz_adv = zeros(Nz, Nx);
    % 1. High-order scheme for the majority of the domain
    dT_dz_adv(1:Nz-2, :) = (-3*T(1:Nz-2, :) + 4*T(2:Nz-1, :) - T(3:Nz, :)) / (2*dz);
    % 2. Fallback to 1st-order at bottom boundaries to prevent index overflow
    dT_dz_adv(Nz-1, :) = (T(Nz, :) - T(Nz-1, :)) / dz; 
    dT_dz_adv(Nz, :)   = (T(Nz, :) - T(Nz-1, :)) / dz; 

    Adv_term = -1 * (Rho_f * Cp_f) .* Vz_map .* dT_dz_adv;
    
    % --- C. Temperature Update Step ---
    dT_dt = (K_map .* (d2T_dx2 + d2T_dz2) + Qr_map + Adv_term) ./ (Rho_map .* Cp_map);
    T = T + dt * dT_dt;
    
    % --- D. Surface Robin Boundary & Water Features ---
    T(1, :) = (h_conv * T_surface + (K_map(1,:)/dz) .* T(2,:)) ./ (h_conv + (K_map(1,:)/dz));
    T(units == 1) = T_surface; % Constant temp for lakes/surface water
    
    % --- E. Convergence Check ---
    if mod(step, 2000) == 0
        err = max(abs(T(:) - T_old(:)));
        if err < Stop_Tol, break; end
        T_old = T;
        if any(isnan(T(:))), error('Numerical divergence! Check dt or physical properties.'); end
        fprintf('Step: %6d | Residual: %.2e | Max Temp: %.1f C\n', step, err, max(T(:)));
    end
end

%% 6. Visualization (16km Core Study Area)
T_plot = T(:, Nx_buffer+1:end);
[X, Z] = meshgrid((0:Nx_core-1)*dx/1000, (0:Nz-1)*dz/1000);

figure('Name','Geothermal Model Results','Position', [100 100 1100 500]);
% Create filled contour plot for background temp field
[~, hf] = contourf(X, Z, T_plot, 40, 'LineStyle', 'none'); hold on;
colormap(jet(256));
c = colorbar; ylabel(c, 'Temperature (^\circC)', 'FontSize', 12);

% Overlay specifically labeled isotherms for feasibility analysis
[C, h_iso] = contour(X, Z, T_plot, [50 70 100 150], 'k', 'ShowText', 'on', 'LineWidth', 1.2);
clabel(C, h_iso, 'FontSize', 10, 'Color', 'w', 'FontWeight', 'bold');

set(gca, 'YDir', 'reverse', 'FontSize', 12);
title('Scottish Highlands Geothermal Feasibility (2nd-Order Upwind)', 'FontSize', 14);
xlabel('Horizontal Distance (km)'); ylabel('Depth (km)');
grid on;

fprintf('Simulation complete. Visuals generated.\n');