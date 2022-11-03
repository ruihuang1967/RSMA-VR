function [w_return_real, w_return_imag, solved_flag] = opt_algo(real_c, imag_c, bf_real, bf_imag, N_input)

    %N = N_in;
    %Nt = Nt_in;
    N = N_input;

    Nt = 6;
    Nr = 1;
    Ni = 2;
    
    FoV_tile = 8;
    tot_tile = FoV_tile*N;
    min_bitrate = 0.00;
    power_max = 1;
    ini_power_ratio = 1.0;

    dis_max = 5;
    dis_min = 0;
    heigh = 3.5;
    IRS_distance = sqrt(dis_max^2+(heigh/2)^2);

    IRS_max = 5;
    IRS_min = 2;
    freq = 60*(10^9);
    sol = 299792458;
    pathloss_op = 2.29;
    bandwidth = 1*(10^9);
    noise_p = 3.9810717055e-21 * bandwidth;
    normalizer = 1/noise_p;

    rate_req = [0, 0];
    %Initializaion of channel
    %channel = rand(Nt,N,1) + 1i*(rand(Nt,N,1));
    trail_num = 100;
    trail_log = zeros(1,1);
    agg_rate_log = zeros(1,1);
    log_rate_log = zeros(1,1);
    user_rate_log = zeros(N,1);

    br_max = 10;
    ini_trail = 1;

    AO_max = 1;
    FP_max = 1;
    ini_MAX = 1;
    penalty_scale = 2;   
    
    channel = real_c + 1i*imag_c;
    %disp(channel)
    
    %generating reflection channel (BS to IRS)
    AoD_IRS = unifrnd(-pi/2,pi/2,1,1);
    IRS_sv_expand = (0:Nt-1).';
    IRS_sv_angle = IRS_sv_expand * sin(AoD_IRS).';
    IRS_sv = exp(1j*IRS_sv_angle);
    H_B_IRS = repmat(IRS_sv,1,Ni)';
    
    IRS_complex_gain = sqrt(2)/2.*(randn(1,1) + 1i*(randn(1,1)));
    H_B_IRS = H_B_IRS .* IRS_complex_gain.* 0;

    %generating reflection channel (IRS to UE)
    H_IRS_U = sqrt(2)/2.*(randn(Ni,N) + 1i*(randn(Ni,N)));
    IRS_UE_dis = unifrnd(IRS_min,IRS_max,N,1) + IRS_distance;
    IRS_UE_path_loss = sqrt((4*pi*IRS_UE_dis*freq./sol).^(-pathloss_op));
    IRS_UE_path_loss = IRS_UE_path_loss .* sqrt(normalizer);
    
    vect_opt = ones(1,Ni);
    Phi = diag(vect_opt);
    H_R = ones(N,Nt,1);
    for n = 1:N
        H_R(n,:) = IRS_UE_path_loss(n,1) .* H_IRS_U(:,n)' * Phi *  H_B_IRS;
    end
   
    
    %Initialization of video tile request
    %cyclic_tile_index = 1:tot_tile;
    %cyclic_tile = repmat(cyclic_tile_index,1,3);
   
    
    %Generate the FoV request based on the real world dataset
    random_seq = randperm(20);
    %user_index = random_seq(1:N); %indeces of the users
    %frame_index = randi([1 video_length-1]);
    FoV_index = zeros(N,FoV_tile);
    for n = 1:N
        FoV_index(n,:) = (n-1)*FoV_tile+1:1:(n)*FoV_tile;
    end
    
    tile_index = zeros(tot_tile,N);
    for n = 1:N
      for i = 1:tot_tile
        tile_index(i,n) = any(FoV_index(n,:) == i);
      end
    end
    
    tile_index_all = sum(tile_index,2);
    tile_index_all_sum = sum(tile_index_all, 'all');
    
    tot_requested_tile = nnz(tile_index_all);
    mostly_shared_tile = nnz(tile_index_all >= 3);
    shared_fraction = mostly_shared_tile/tot_requested_tile;
    
    
    c_v = ones(1,N)./N;
    br_v = zeros(tot_tile, N);
    p_v = zeros(tot_tile, N);
    for n = 1:N
        sum_tile_n = sum(tile_index(:,n), 'all');
        p_v(:,n) = tile_index(:,n)/sum_tile_n;
        br_v(:,n) = tile_index(:,n).*min_bitrate;
    end
        
    %Initialization of beamforming vector
    P = bf_real + 1i*bf_imag;
    %disp(P);

    obj_end_ite = zeros(1,1);
    obj_old = 0;   
    feasible_flag = 0;
    sum_rate_old = 0;
    for AO_counter = 1:AO_max
        %% FP Part              
        %Initialization of y vector (auxi for private rate)
        y = zeros(N,1);
        for n = 1:N
            interf_n = 0.;
            H_n = channel(n,:) + H_R(n,:);
            tx_bfv_n = P(:,n+1);
            for j = 1:N
                if j ~= n
                    tx_bfv_j = P(:,j+1);
                    interf_n = interf_n + H_n * tx_bfv_j * tx_bfv_j' * H_n';
                end
            end
            interf_n = interf_n + 1 * eye(1);
            y(n,1)= interf_n^(-1) * H_n * tx_bfv_n;
        end

        %Initialization of x vector (auxi for common rate)
        x = zeros(N,1);
        tx_bfv_c = P(:,1);
        for n = 1:N
            interf_n = 0.;
            H_n = channel(n,:) + H_R(n,:);
            for j = 1:N
                tx_bfv_j = P(:,j+1);
                interf_n = interf_n + H_n * tx_bfv_j * tx_bfv_j' * H_n';
            end
            interf_n = interf_n + 1 * eye(1);
            x(n,1)= interf_n^(-1) * H_n * tx_bfv_c;
        end
        %disp(y);
        if isnan(y) | isnan(x)
            break;
        end
        %%
        %CVX Parts for Beamforming
        cvx_begin quiet
        cvx_expert true
        cvx_solver mosek
          variable P(Nt,N+1) complex
          variable R_c nonnegative
          expression r_c_relax(N)
          expression rate_per_tile(tot_tile, N)
          expression r_p_relax(N)
          obj = 0;
          for n = 1:N
            interf_n = 0.;
            interf_c = 0.;
            H_n = channel(n,:) + H_R(n,:);
            tx_bfv_n = P(:,n+1);
            tx_bfv_c = P(:,1);
            for j = 1:N
                tx_bfv_j = P(:,j+1);
                if j ~= n                  
                    interf_n = real(interf_n + quad_form(H_n * tx_bfv_j, eye(1)));
                end
                interf_c = real(interf_c + quad_form(H_n * tx_bfv_j, eye(1)));
            end
            interf_n = interf_n + 1 * eye(1);
            interf_c = interf_c + 1 * eye(1);
            y_n = y(n,1);
            x_n = x(n,1);
            cm_r_n = R_c.*c_v(1,n);
            %disp(interf_n)
            r_p_relax(n) = log(1 + 2*real(y_n' * H_n * tx_bfv_n) -  y_n'*y_n*interf_n);
            obj = obj + log(1 + 2*real(y_n' * H_n * tx_bfv_n) -  y_n'*y_n*interf_n) + cm_r_n;
            r_c_relax(n) = log(1 + 2*real(x_n' * H_n * tx_bfv_c) -  x_n'*x_n*interf_c);
          end
          maximize(obj)
          subject to
            square_pos(norm(P,'fro'))<=1;
            r_c_relax - R_c >= 0;

        cvx_end

        if isnan(R_c)
            break;
        end  

        %Rate check
        p_rate_n = zeros(N,1);
        common_rate_n = zeros(N,1);
        all_rate_n = zeros(1,N);
        tx_bfv_c = P(:,1);
        for n = 1:N
            itf_n = 0.;
            H_n = channel(n,:) + H_R(n,:);
            tx_bfv_n = P(:,n+1);
            for j =1:N
                if j~=n
                    tx_bfv_j = P(:,j+1);
                    itf_n = itf_n + abs(H_n * tx_bfv_j)^2;
                end
            end
            sinr = abs(H_n * tx_bfv_n)^2/(itf_n+1);
            sinr_c = abs(H_n * tx_bfv_c)^2/(abs(H_n * tx_bfv_n)^2+itf_n+1);
            p_rate_n(n,1) = log(1+sinr)/log(2);
            common_rate_n(n,1) = log(1+sinr_c)/log(2);
        end
        R_c_check = min(common_rate_n);
        for n = 1:N
            all_rate_n(1,n) = p_rate_n(n,1) + c_v(1,n) .*R_c_check;
        end
        sum_rate_new = sum(all_rate_n, 'all');
        %fprintf('End of %d ite in FP: sum_rate_new %f \n',fp_counter, sum_rate_new);
        x=1;                          

        if sum_rate_new - sum_rate_old<=0.01
            check_flag =1;
            break;
        else
            sum_rate_old = sum_rate_new;
        end  
    end           

     %fprintf('End of %d ite in AO: %f \n',AO_counter, obj_new);
    P = full(P);
    if feasible_flag == 0
        w_return_real = real(P);
        %disp(w_return_real)
        w_return_imag = imag(P);
        %disp(w_return_imag)
        solved_flag = 0;
    else 
        w_return_real = real(P);
        %disp(w_return_real)
        w_return_imag = imag(P);
        %disp(w_return_imag)
        solved_flag = 1;
    end
    %disp(sum_rate_new);
end

