from unsupervised_tools.unsupervised_utils import split_every, clamp_params, get_fake_obs_via_ECC, \
    generation_step_and_save
import torch_geometric
import torch


def train_plain_gan(max_num_of_epochs, batch_size, list_of_observations, line_graphs_for_list_of_observations,
                    dataset, device, N_CRITIC_STEPS, critic, ECC_nodes, ECC_edges, optimizer_critic,
                    optimizer_nodes, optimizer_edges, critic_loss_log, generator_loss_log,
                    rnn, output, absence_net, args, qm9_smiles):

    for epoch in range(max_num_of_epochs):
        # have to reset at each epoch, otherwise they run out and training stops
        iter_fake_graphs = split_every(batch_size, list_of_observations)
        iter_fake_line_graphs = split_every(batch_size, line_graphs_for_list_of_observations)
        iter_real_data = split_every(batch_size, dataset)

        crit_steps = 0

        for i in range(len(list_of_observations) // batch_size):

            real_batch_1 = next(iter_real_data)
            real_batch_2 = torch_geometric.data.DataLoader(real_batch_1, batch_size=batch_size, shuffle=True)
            real_batch_3 = iter(real_batch_2)
            real_batch_4 = real_batch_3.next()
            real_batch_4.to(device)

            nodes_batch_1 = next(iter_fake_graphs)
            edges_batch_1 = next(iter_fake_line_graphs)

            if crit_steps < N_CRITIC_STEPS:
                critic.train()
                critic.zero_grad()

                ECC_nodes.eval()
                ECC_edges.eval()

                clamp_params(critic)
                err_real = torch.mean(critic(real_batch_4))  # E[D(x)]

                fake_datalist = []
                for g_nodes, g_edges in zip(nodes_batch_1, edges_batch_1):
                    fake_datalist.append(get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges, epoch))

                fake_dataloader_pyg = torch_geometric.data.DataLoader(fake_datalist, batch_size=batch_size, shuffle=True)
                fake_data_iterator = iter(fake_dataloader_pyg)
                batch_generated = fake_data_iterator.next()
                batch_generated.to(device)

                err_fake = torch.mean(critic(batch_generated))  # E[D(G(z))]

                critic_loss = err_fake - err_real  # want this min
                critic_loss.backward()  # retain_graph=True
                optimizer_critic.step()
                crit_steps += 1

            else:
                ECC_nodes.train()
                ECC_edges.train()
                ECC_nodes.zero_grad()
                ECC_edges.zero_grad()
                critic.eval()

                fake_datalist = []
                for g_nodes, g_edges in zip(nodes_batch_1, edges_batch_1):
                    fake_datalist.append(get_fake_obs_via_ECC(ECC_nodes, ECC_edges, g_nodes, g_edges, epoch))

                fake_dataloader_pyg = torch_geometric.data.DataLoader(fake_datalist, batch_size=batch_size, shuffle=True)
                fake_data_iterator = iter(fake_dataloader_pyg)
                batch_generated = fake_data_iterator.next()
                batch_generated.to(device)

                output_critic_fake = critic(batch_generated)
                generator_loss = -torch.mean(output_critic_fake)
                generator_loss.backward()

                optimizer_nodes.step()
                optimizer_edges.step()
                crit_steps = 0

                critic_loss_log.info(str(epoch) + ' , ' + str(critic_loss.item()))
                generator_loss_log.info(str(epoch) + ' , ' + str(generator_loss.item()))

                print(
                    f"Epoch: {epoch}/{max_num_of_epochs}, batch: {i}/{len(list_of_observations) // batch_size}, Err_real - Err_fake = {critic_loss}, temp: {500 / (epoch + 1)}")

        generation_step_and_save(epoch, ECC_nodes, ECC_edges, critic, rnn, output, absence_net, device,
                                 args,
                                 qm9_smiles)
