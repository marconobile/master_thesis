import os
import torch
from model import GRU_plain  # if not triple head
from args import Args
from get_generator  import get_generator


# from model_triple import GRU_plain

import os
import torch
from model import GRU_plain  # if not triple head
from args import Args
from get_generator  import get_generator
# from disc_model import Reward_Net_Joint


# from model_triple import GRU_plain


def save(epoch, rnn, output, absence_net):
    # save(epoch, rnn, optimizer_rnn, scheduler_rnn, output, optimizer_output, scheduler_output):
    path = os.getcwd() + "/weights/"  # "/content/gdrive/My Drive/GraphRNN_weights/"
    checkpoint_rnn = {
        'model_state_dict': rnn.state_dict(),
        # 'epoch': epoch,
        # 'optimizer_rnn': optimizer_rnn.state_dict(),
        # 'scheduler_rnn': scheduler_rnn.state_dict()
    }

    torch.save(checkpoint_rnn, path + f'nodeRNN_checkpoint_{epoch}.pth')

    checkpoint_output = {
        'model_state_dict': output.state_dict(),
        # 'epoch': epoch,
        # 'optimizer_output': optimizer_output.state_dict(),
        # 'scheduler_output': scheduler_output.state_dict()
    }

    torch.save(checkpoint_output, path + f'edgeRNN_checkpoint_{epoch}.pth')

    if absence_net:
        checkpoint_absence = {
            'model_state_dict': absence_net.state_dict(),
            # 'epoch': epoch,
            # 'optimizer_output': optimizer_output.state_dict(),
            # 'scheduler_output': scheduler_output.state_dict()
        }

        torch.save(checkpoint_absence, path + f'absenceRNN_checkpoint_{epoch}.pth')


###############################################################################################################################################################

def load_reward_net(to_be_loaded):
    path = "./weights"
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")

    reward_net = Reward_Net_Joint()

    PATH = path + f'/reward_net_checkpoint_{to_be_loaded}.pth'
    if cuda:
        checkpoint1 = torch.load(PATH)
    else:
        checkpoint1 = torch.load(PATH, map_location='cpu')
    reward_net.load_state_dict(checkpoint1['model_state_dict'])
    reward_net.train()
    return reward_net


def load_for_gpu(to_be_loaded):  # to_be_loaded = int(last_epoch)
    path = "./weights"
    cuda = True if torch.cuda.is_available() else False
    device = torch.device("cuda:0" if cuda else "cpu")


    args = Args()
    if args.pretrained:

        if args.with_absence_net:
            rnn, output, absence_net = get_generator()

            PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
            if cuda:
                checkpoint1 = torch.load(PATH1)
            else:
                checkpoint1 = torch.load(PATH1, map_location='cpu')
            rnn.load_state_dict(checkpoint1['model_state_dict'])
            rnn.train()

            PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
            if cuda:
                checkpoint2 = torch.load(PATH2)
            else:
                checkpoint2 = torch.load(PATH2, map_location='cpu')
            output.load_state_dict(checkpoint2['model_state_dict'])
            output.train()

            PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
            if cuda:
                checkpoint3 = torch.load(PATH3)
            else:
                checkpoint3 = torch.load(PATH3, map_location = 'cpu')
            absence_net.load_state_dict(checkpoint3['model_state_dict'])
            absence_net.train()

            rnn.to(device)
            output.to(device)
            absence_net.to(device)
            return rnn, output, absence_net

    elif ((args.with_absence_net == True) and (args.supervised ==  False)):
        rnn = False
        output = False
        _, _, absence_net = get_generator()
        PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
        if cuda:
            checkpoint3 = torch.load(PATH3)
        else:
            checkpoint3 = torch.load(PATH3, map_location='cpu')
        absence_net.load_state_dict(checkpoint3['model_state_dict'])
        absence_net.train()
        absence_net.to(device)

        return rnn, output, absence_net


# def load_for_CPU(to_be_loaded):  # to_be_loaded = int(last_epoch)
#     path = "./weights"
#
#     num_layers = 4
#
#     embedding_size_rnn = 128
#     hidden_size_rnn = 128
#
#     hidden_size_rnn_output = 64
#     embedding_size_rnn_output = 64
#
#     rnn = GRU_plain(input_size=44, embedding_size=embedding_size_rnn,
#                     hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
#                     has_output=True, output_size=hidden_size_rnn_output, node_lvl=True, out_middle_layer=64)
#
#     absence_net = GRU_plain(input_size=44, embedding_size=64,
#                             hidden_size=64, num_layers=4, has_input=True,
#                             has_output=True, output_size=8, node_lvl=False, out_middle_layer=64)
#
#     output = GRU_plain(input_size=5, embedding_size=embedding_size_rnn_output,
#                        hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
#                        has_output=True, output_size=4, node_lvl=False, out_middle_layer=32)  # output_size=4
#
#     PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
#     checkpoint1 = torch.load(PATH1, map_location='cpu')
#     rnn.load_state_dict(checkpoint1['model_state_dict'])
#     rnn.train()
#
#     PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
#     checkpoint2 = torch.load(PATH2, map_location='cpu')
#     output.load_state_dict(checkpoint2['model_state_dict'])
#     output.train()
#
#     PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
#     checkpoint3 = torch.load(PATH3, map_location='cpu')
#     absence_net.load_state_dict(checkpoint3['model_state_dict'])
#     absence_net.train()
#
#     return rnn, output, absence_net

# def resume_training(epoch):
#     rnn, output = load_for_gpu(epoch)
#     counter_test = 0 # FIX THIS
#     max_epoch = 1000 #3000
#     while epoch <= max_epoch:
#         print(f'epoch {epoch} starts')
#         loss_this_epoch = train_rnn_epoch(epoch = epoch, rnn = rnn, output = output, data_loader = train_set,
#                     optimizer_rnn = optimizer_rnn, optimizer_output = optimizer_output, max_num_of_epochs = max_epoch, writer = writer)
#         # print(f'epoch {epoch} over')

#         writer.add_scalar('avg(Loss)xepoch', loss_this_epoch, epoch)

#         if epoch % 5 == 0:
#             test_loss = test_rnn_single_epoch(epoch = epoch, rnn = rnn, output = output, data_loader = test_set,
#                     optimizer_rnn = optimizer_rnn, optimizer_output = optimizer_output, max_num_of_epochs = max_epoch, writer = writer)
#             writer.add_scalar('Test loss', test_loss, counter_test)
#             print(f'Evaluation step number {counter_test} on test data, loss value: {test_val}')
#             counter_test+=1

#         if epoch % 10 == 0:
#             save()
#             print(f'model saved at epoch {epoch} !')

#         if epoch % 5 == 0:
#             print(f'Epoch: {epoch}/{max_epoch}, avg Loss: {loss_this_epoch}')
#         epoch += 1

#     writer.close()


# def load_and_inference(epoch):
#     rnn, output = load_for_CPU(epoch)
#     print("loading...")
#     print(rnn)
#     print('##'*20)
#     print(output)
#     print("done!")

#     return rnn, output



















#
# def save(epoch, rnn, output, absence_net):
#     # save(epoch, rnn, optimizer_rnn, scheduler_rnn, output, optimizer_output, scheduler_output):
#     path = os.getcwd() + "/weights/"  # "/content/gdrive/My Drive/GraphRNN_weights/"
#     checkpoint_rnn = {
#         'model_state_dict': rnn.state_dict(),
#         # 'epoch': epoch,
#         # 'optimizer_rnn': optimizer_rnn.state_dict(),
#         # 'scheduler_rnn': scheduler_rnn.state_dict()
#     }
#
#     torch.save(checkpoint_rnn, path + f'nodeRNN_checkpoint_{epoch}.pth')
#
#     checkpoint_output = {
#         'model_state_dict': output.state_dict(),
#         # 'epoch': epoch,
#         # 'optimizer_output': optimizer_output.state_dict(),
#         # 'scheduler_output': scheduler_output.state_dict()
#     }
#
#     torch.save(checkpoint_output, path + f'edgeRNN_checkpoint_{epoch}.pth')
#
#     if absence_net:
#         checkpoint_absence = {
#             'model_state_dict': absence_net.state_dict(),
#             # 'epoch': epoch,
#             # 'optimizer_output': optimizer_output.state_dict(),
#             # 'scheduler_output': scheduler_output.state_dict()
#         }
#
#         torch.save(checkpoint_absence, path + f'absenceRNN_checkpoint_{epoch}.pth')
#
#
# ###############################################################################################################################################################
#
# def load_for_gpu(to_be_loaded):  # to_be_loaded = int(last_epoch)
#     path = "./weights"
#     cuda = True if torch.cuda.is_available() else False
#     device = torch.device("cuda:0" if cuda else "cpu")
#
#
#     args = Args()
#     if args.pretrained:
#
#         if args.with_absence_net:
#             rnn, output, absence_net = get_generator()
#
#             PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
#             checkpoint1 = torch.load(PATH1)
#             rnn.load_state_dict(checkpoint1['model_state_dict'])
#             rnn.train()
#
#             PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
#             checkpoint2 = torch.load(PATH2)
#             output.load_state_dict(checkpoint2['model_state_dict'])
#             output.train()
#
#             PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
#             checkpoint3 = torch.load(PATH3)
#             absence_net.load_state_dict(checkpoint3['model_state_dict'])
#             absence_net.train()
#
#             rnn.to(device)
#             output.to(device)
#             absence_net.to(device)
#             return rnn, output, absence_net
#
#     elif ((args.with_absence_net == True) and (args.gan ==  True)):
#         rnn = False
#         output = False
#         _, _, absence_net = get_generator()
#         PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
#         checkpoint3 = torch.load(PATH3)
#         absence_net.load_state_dict(checkpoint3['model_state_dict'])
#         absence_net.train()
#         absence_net.to(device)
#
#         return rnn, output, absence_net
#
#
# # def load_for_CPU(to_be_loaded):  # to_be_loaded = int(last_epoch)
# #     path = "./weights"
# #
# #     num_layers = 4
# #
# #     embedding_size_rnn = 128
# #     hidden_size_rnn = 128
# #
# #     hidden_size_rnn_output = 64
# #     embedding_size_rnn_output = 64
# #
# #     rnn = GRU_plain(input_size=44, embedding_size=embedding_size_rnn,
# #                     hidden_size=hidden_size_rnn, num_layers=num_layers, has_input=True,
# #                     has_output=True, output_size=hidden_size_rnn_output, node_lvl=True, out_middle_layer=64)
# #
# #     absence_net = GRU_plain(input_size=44, embedding_size=64,
# #                             hidden_size=64, num_layers=4, has_input=True,
# #                             has_output=True, output_size=8, node_lvl=False, out_middle_layer=64)
# #
# #     output = GRU_plain(input_size=5, embedding_size=embedding_size_rnn_output,
# #                        hidden_size=hidden_size_rnn_output, num_layers=num_layers, has_input=True,
# #                        has_output=True, output_size=4, node_lvl=False, out_middle_layer=32)  # output_size=4
# #
# #     PATH1 = path + f'/nodeRNN_checkpoint_{to_be_loaded}.pth'
# #     checkpoint1 = torch.load(PATH1, map_location='cpu')
# #     rnn.load_state_dict(checkpoint1['model_state_dict'])
# #     rnn.train()
# #
# #     PATH2 = path + f'/edgeRNN_checkpoint_{to_be_loaded}.pth'
# #     checkpoint2 = torch.load(PATH2, map_location='cpu')
# #     output.load_state_dict(checkpoint2['model_state_dict'])
# #     output.train()
# #
# #     PATH3 = path + f'/absenceRNN_checkpoint_{to_be_loaded}.pth'  # absenceRNN_checkpoint_
# #     checkpoint3 = torch.load(PATH3, map_location='cpu')
# #     absence_net.load_state_dict(checkpoint3['model_state_dict'])
# #     absence_net.train()
# #
# #     return rnn, output, absence_net
#
# # def resume_training(epoch):
# #     rnn, output = load_for_gpu(epoch)
# #     counter_test = 0 # FIX THIS
# #     max_epoch = 1000 #3000
# #     while epoch <= max_epoch:
# #         print(f'epoch {epoch} starts')
# #         loss_this_epoch = train_rnn_epoch(epoch = epoch, rnn = rnn, output = output, data_loader = train_set,
# #                     optimizer_rnn = optimizer_rnn, optimizer_output = optimizer_output, max_num_of_epochs = max_epoch, writer = writer)
# #         # print(f'epoch {epoch} over')
#
# #         writer.add_scalar('avg(Loss)xepoch', loss_this_epoch, epoch)
#
# #         if epoch % 5 == 0:
# #             test_loss = test_rnn_single_epoch(epoch = epoch, rnn = rnn, output = output, data_loader = test_set,
# #                     optimizer_rnn = optimizer_rnn, optimizer_output = optimizer_output, max_num_of_epochs = max_epoch, writer = writer)
# #             writer.add_scalar('Test loss', test_loss, counter_test)
# #             print(f'Evaluation step number {counter_test} on test data, loss value: {test_val}')
# #             counter_test+=1
#
# #         if epoch % 10 == 0:
# #             save()
# #             print(f'model saved at epoch {epoch} !')
#
# #         if epoch % 5 == 0:
# #             print(f'Epoch: {epoch}/{max_epoch}, avg Loss: {loss_this_epoch}')
# #         epoch += 1
#
# #     writer.close()
#
#
# # def load_and_inference(epoch):
# #     rnn, output = load_for_CPU(epoch)
# #     print("loading...")
# #     print(rnn)
# #     print('##'*20)
# #     print(output)
# #     print("done!")
#
# #     return rnn, output
