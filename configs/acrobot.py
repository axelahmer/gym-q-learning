class config():
    # env config
    env_name = 'Acrobot-v1'
    render_train = False
    render_test = True
    test_episodes = 5

    # q_model structure
    hidden_units = 32
    buffer_size = 50_000

    # output config
    output_path = "results/acrobot/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # model and training config
    num_episodes_test = 20
    saving_freq = 5000
    log_freq = 50
    eval_freq = 3000
    batch_size = 32

    # hyper params
    train_nsteps = 10_000
    gamma = 0.99
    lr_begin = 0.01
    lr_end = 0.001
    lr_nsteps = train_nsteps * 0.5
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = train_nsteps * 0.9
