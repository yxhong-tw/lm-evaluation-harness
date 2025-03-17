import torch
import torch.nn.functional as F


class SimilarityCalculator():
    """ The SimilarityCalculator class.
    """

    def __init__(self) -> None:
        """ Initialize the similarity calculator.
        """

        try:
            from ...logging import get_logger
            from ...utils import get_path

            self.logger = get_logger()
            self.source = f'{get_path(source_file=__file__)}.{SimilarityCalculator.__name__}'
        except:
            pass

    def __call__(
        self,
        similarity_type: str,
        x1: torch.Tensor,
        x2: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """ Calculate the similarity between two tensors.

        Args:
            similarity_type (str): The type of similarity.
            x1 (torch.Tensor): The first tensor.
            x2 (torch.Tensor): The second tensor.

        Raises:
            ValueError: The similarity type is NOT supported.

        Returns:
            torch.Tensor: The similarity.
        """

        match similarity_type:
            case 'cosine':
                similarity_calculator = F.cosine_similarity

                dim = kwargs['dim'] if 'dim' in kwargs.keys() else 1
                eps = kwargs['eps'] if 'eps' in kwargs.keys() else 1e-8

                similarity = similarity_calculator(
                    x1=x1,
                    x2=x2,
                    dim=dim,
                    eps=eps,
                )
            case 'euclidean':
                similarity_calculator = F.pairwise_distance

                p = kwargs['p'] if 'p' in kwargs.keys() else 2
                eps = kwargs['eps'] if 'eps' in kwargs.keys() else 1e-6
                keepdim = kwargs['keepdim'] if 'keepdim' in kwargs.keys(
                ) else False

                similarity = similarity_calculator(
                    x1=x1,
                    x2=x2,
                    p=p,
                    eps=eps,
                    keepdim=keepdim,
                )
            case _:
                message = f'The similarity type {similarity_type} is NOT supported.'

                try:
                    self.logger.log(
                        message=message,
                        level='error',
                        source=self.source,
                    )
                except:
                    print(message)

                    pass

                raise ValueError(message)

        return similarity
