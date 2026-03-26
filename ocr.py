import cv2
import numpy as np
import tensorflow as tf


class OCR:
    """
    Character-level OCR using a frozen TensorFlow protobuf model.

    Parameters
    ----------
    modelFile : str
        Path to the frozen .pb graph file.
    labelFile : str
        Path to the plain-text label file (one label per line).
    """

    def __init__(self, modelFile: str, labelFile: str):
        self.model_file = modelFile
        self.label_file = labelFile
        self.label = self._load_labels(labelFile)
        self.graph = self._load_graph(modelFile)
        self.sess = tf.compat.v1.Session(
            graph=self.graph,
            config=tf.compat.v1.ConfigProto()
        )

    # ------------------------------------------------------------------
    # Model / label loading
    # ------------------------------------------------------------------

    def _load_graph(self, model_file: str) -> tf.Graph:
        """Load a frozen TensorFlow graph from a .pb file."""
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()

        with open(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())

        with graph.as_default():
            tf.import_graph_def(graph_def)

        return graph

    def _load_labels(self, label_file: str) -> list:
        """Load labels from a plain-text file, one label per line."""
        labels = []
        for line in tf.io.gfile.GFile(label_file).readlines():
            labels.append(line.rstrip())
        return labels

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def convert_tensor(self, image: np.ndarray, output_size: int) -> np.ndarray:
        """
        Resize *image* to (output_size × output_size), normalise to
        [-0.5, 0.5], and add a batch dimension.
        """
        image = cv2.resize(
            image,
            dsize=(output_size, output_size),
            interpolation=cv2.INTER_CUBIC
        )
        np_image = np.asarray(image, dtype='float32')
        np_image = cv2.normalize(np_image, None, -0.5, 0.5, cv2.NORM_MINMAX)
        return np.expand_dims(np_image, axis=0)

    def label_image(self, tensor: np.ndarray) -> str:
        """Run a single tensor through the model and return the top label."""
        input_op  = self.graph.get_operation_by_name("import/input")
        output_op = self.graph.get_operation_by_name("import/final_result")

        results = self.sess.run(
            output_op.outputs[0],
            {input_op.outputs[0]: tensor}
        )
        results = np.squeeze(results)
        top_index = results.argsort()[-1]
        return self.label[top_index]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_image_list(self, list_images: list, image_size: int = 128):
        """
        Recognise a list of character-image crops and concatenate the
        predicted labels into a plate string.

        Parameters
        ----------
        list_images : list of np.ndarray
            Individual character crops (BGR).
        image_size : int
            Target size fed to the model (default 128).

        Returns
        -------
        plate : str
            Concatenated recognised characters.
        length : int
            Number of characters recognised (0 means nothing found).
        """
        plate = ""
        for img in list_images:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
            tensor = self.convert_tensor(img, image_size)
            plate += self.label_image(tensor)

        return plate, len(plate)