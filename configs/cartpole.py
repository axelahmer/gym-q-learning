class config():

    # env config
    env_name = 'CartPole-v1'
    render_train = False
    render_test = True
    train_nsteps = 20_000
    buffer_size = 5_000

    # q_model structure
    hidden_units = 64

    # output config
    output_path = "results/cartpole/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"

    # model and training config

    num_episodes_test = 10
    num_episodes_eval = 5
    saving_freq = 5000
    eval_freq = 3000
    batch_size = 32

    # hyper params

    gamma = 0.99
    lr_begin = 0.1
    lr_end = 0.05
    lr_nsteps = train_nsteps * 0.8
    eps_begin = 1
    eps_end = 0.01
    eps_nsteps = train_nsteps * 0.8
