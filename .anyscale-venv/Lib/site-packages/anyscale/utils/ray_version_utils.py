from anyscale.shared_anyscale_utils.utils.ray_semver import ray_semver_compare


def get_correct_name_for_base_image_name(base_image_name: str) -> str:
    """
    Modify base_image_name to be a correct base image name if it is not.

    This function checks the base_image_name, if it looks correct, returns it.
    If not, try to modify it to be a correct base image name. This can be used
    as a hint for users on what they might want to use when the given name
    is not a right one.

    Args:
        base_image_name (str): The name of the base image to be checked.

    Returns:
        str: The modified base image name based on version and naming conventions.

    Example:
        To check a base image name, you can call this function like this:
        >>> get_correct_name_for_base_image_name("anyscale/ray:2.7.0-py38-cuda121"), it should be modified to "anyscale/ray:2.7.0optimized-py38-cuda121"
    """
    # to extract "2.7.0", "2.7.0oss", "2.7.0optimized", "2.0.1rc" as ray_version
    ray_version = get_ray_version(base_image_name)

    # If it has an explicit suffix, respect it.
    if ray_version.endswith(("optimized", "oss")) or "rc" in ray_version:
        return base_image_name

    # Ray 2.7.x images do not exist on anyscale. Only "optimized" version exists.
    if (
        ray_semver_compare(ray_version, "2.7.0") >= 0
        and ray_semver_compare(ray_version, "2.8.0") < 0
    ):  # ray_version ~= 2.7.?
        return base_image_name.replace(ray_version, f"{ray_version}optimized")

    return base_image_name


def get_ray_version(base_image: str) -> str:
    """Returns the Ray version in use based on the base image.

    Args:
        base_image: e.g. anyscale/ray-ml:1.9.0-cpu

    Returns:
        The ray version, e.g. 1.9.0.
    """
    # e.g. 1.9.0-cpu
    image_version = base_image.split(":")[-1]
    # e.g. 1.9.0
    ray_version = image_version.split("-")[0]
    return ray_version
