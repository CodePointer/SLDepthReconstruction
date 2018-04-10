#include "reconstructor.h"

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_log_dir = "./log/";
  FLAGS_logbufsecs = 0;
  google::InstallFailureSignalHandler();

  Reconstructor my_recon;
  bool status = true;

  if (status) {
    status = my_recon.Init();
  }
  if (status) {
    status = my_recon.Run();
  }
  if (status) {
    status = my_recon.Close();
  }
  if (!status) {
    LOG(ERROR) << "Main function closed error.";
    // ErrorThrow("Main function closed Error.");
  }
  return 0;
}